# coding: utf-8
import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
import yaml
from loguru import logger
from omegaconf import OmegaConf
from torchvision.transforms import GaussianBlur

from smooth_network import SmoothNet, build_SmoothNet
from spatial_network import SpatialNet, build_SpatialNet
from sv_comp.dataset import MultiWarpDataset
from temporal_network import TemporalNet, build_TemporalNet
import utils.torch_tps_transform as torch_tps_transform
import utils.torch_tps_transform_point as torch_tps_transform_point

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    def compare_psnr(img1, img2, data_range):
        return peak_signal_noise_ratio(img1, img2, data_range=data_range)

    def compare_ssim(img1, img2, data_range):
        return structural_similarity(img1, img2, data_range=data_range, channel_axis=-1)
except ImportError:
    import skimage.measure

    def compare_psnr(img1, img2, data_range):
        return skimage.measure.compare_psnr(img1, img2, data_range)

    def compare_ssim(img1, img2, data_range):
        return skimage.measure.compare_ssim(img1, img2, data_range=data_range, multichannel=True)


last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = os.path.join(last_path, 'model')

grid_h = 6
grid_w = 8


def linear_blender(ref, tgt, ref_m, tgt_m, mask=False):
    blur = GaussianBlur(kernel_size=(21, 21), sigma=20)
    device = ref.device

    if torch.count_nonzero(ref_m) == 0 or torch.count_nonzero(tgt_m) == 0:
        if mask:
            return ref_m.clamp(0, 1)
        return ref * ref_m + tgt * tgt_m

    r1, c1 = torch.nonzero(ref_m[0, 0], as_tuple=True)
    r2, c2 = torch.nonzero(tgt_m[0, 0], as_tuple=True)
    center1 = (r1.float().mean(), c1.float().mean())
    center2 = (r2.float().mean(), c2.float().mean())
    vec = (center2[0] - center1[0], center2[1] - center1[1])

    ovl = (ref_m * tgt_m).round()[:, 0].unsqueeze(1)
    ref_m_ = ref_m[:, 0].unsqueeze(1) - ovl
    if torch.count_nonzero(ovl) == 0:
        mask1 = blur(ref_m).clamp(0, 1)
        if mask:
            return mask1
        mask2 = (1 - mask1) * tgt_m
        return ref * mask1 + tgt * mask2

    r, c = torch.nonzero(ovl[0, 0], as_tuple=True)
    ovl_mask = torch.zeros_like(ref_m_, device=device)
    proj_val = (r - center1[0]) * vec[0] + (c - center1[1]) * vec[1]
    ovl_mask[ovl.bool()] = (proj_val - proj_val.min()) / (proj_val.max() - proj_val.min() + 1e-3)

    mask1 = (blur(ref_m_ + (1 - ovl_mask) * ref_m[:, 0].unsqueeze(1)) * ref_m + ref_m_).clamp(0, 1)
    if mask:
        return mask1

    mask2 = (1 - mask1) * tgt_m
    return ref * mask1 + tgt * mask2


def recover_mesh(norm_mesh, height, width):
    batch_size = norm_mesh.size()[0]
    mesh_w = (norm_mesh[..., 0] + 1) * float(width) / 2.0
    mesh_h = (norm_mesh[..., 1] + 1) * float(height) / 2.0
    mesh = torch.stack([mesh_w, mesh_h], 2)
    return mesh.reshape([batch_size, grid_h + 1, grid_w + 1, 2])


def get_rigid_mesh(batch_size, height, width, device):
    ww = torch.matmul(
        torch.ones([grid_h + 1, 1], device=device),
        torch.unsqueeze(torch.linspace(0.0, float(width), grid_w + 1, device=device), 0),
    )
    hh = torch.matmul(
        torch.unsqueeze(torch.linspace(0.0, float(height), grid_h + 1, device=device), 1),
        torch.ones([1, grid_w + 1], device=device),
    )
    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)), 2)
    return ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)


def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[..., 0] * 2.0 / float(width) - 1.0
    mesh_h = mesh[..., 1] * 2.0 / float(height) - 1.0
    norm_mesh = torch.stack([mesh_w, mesh_h], 3)
    return norm_mesh.reshape([batch_size, -1, 2])


def get_stable_sqe(img1_list, img2_list, ori_mesh, warp_mode='FAST', fusion_mode='AVERAGE'):
    device = ori_mesh.device
    batch_size, _, img_h, img_w = img2_list[0].shape

    # The network predicts meshes on 360x480 inputs. Convert them to the
    # current image resolution before warping full-resolution images.
    mesh = torch.stack([ori_mesh[..., 0] * img_w / 480.0, ori_mesh[..., 1] * img_h / 360.0], 4)

    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w, device)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)

    width_max = torch.maximum(torch.tensor(float(img_w), device=device), torch.max(mesh[..., 0]))
    width_min = torch.minimum(torch.tensor(0.0, device=device), torch.min(mesh[..., 0]))
    height_max = torch.maximum(torch.tensor(float(img_h), device=device), torch.max(mesh[..., 1]))
    height_min = torch.minimum(torch.tensor(0.0, device=device), torch.min(mesh[..., 1]))

    width_min_int = int(torch.floor(width_min).item())
    width_max_int = int(torch.ceil(width_max).item())
    height_min_int = int(torch.floor(height_min).item())
    height_max_int = int(torch.ceil(height_max).item())
    out_width = width_max_int - width_min_int + 1
    out_height = height_max_int - height_min_int + 1

    stable_list = []
    for i in range(len(img2_list)):
        mesh_frame = mesh[:, i, :, :, :]
        mesh_trans = torch.stack([mesh_frame[..., 0] - width_min_int, mesh_frame[..., 1] - height_min_int], 3)
        norm_mesh = get_norm_mesh(mesh_trans, out_height, out_width)

        img1 = (img1_list[i].to(device) + 1) * 127.5
        img2 = (img2_list[i].to(device) + 1) * 127.5
        img1_warp = torch.zeros((1, 3, out_height, out_width), device=device)
        img1_warp[:, :, -height_min_int:-height_min_int + img_h, -width_min_int:-width_min_int + img_w] = img1

        img2_warp = torch_tps_transform.transformer(img2, norm_mesh, norm_rigid_mesh, (out_height, out_width), mode=warp_mode)
        if fusion_mode == 'LINEAR':
            mask1 = torch.zeros((1, 1, out_height, out_width), device=device)
            mask1[:, :, -height_min_int:-height_min_int + img_h, -width_min_int:-width_min_int + img_w] = 1
            mask2 = torch_tps_transform.transformer(
                torch.ones_like(img2[:, :1, ...], device=device), norm_mesh, norm_rigid_mesh, (out_height, out_width), mode=warp_mode
            )
            fusion = linear_blender(img1_warp, img2_warp, mask1, mask2)[0]
        else:
            fusion = img1_warp[0] * (img1_warp[0] / (img1_warp[0] + img2_warp[0] + 1e-6))
            fusion += img2_warp[0] * (img2_warp[0] / (img1_warp[0] + img2_warp[0] + 1e-6))

        stable_list.append(fusion.detach().cpu().numpy().transpose(1, 2, 0))

    return stable_list, out_width, out_height


def get_stable_sqe_metric(img2_list, ori_mesh, warp_mode='NORMAL'):
    device = ori_mesh.device
    batch_size, _, img_h, img_w = img2_list[0].shape

    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w, device)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)

    stable_list = []
    for i in range(len(img2_list)):
        mesh = ori_mesh[:, i, :, :, :]
        norm_mesh = get_norm_mesh(mesh, img_h, img_w)
        img2 = (img2_list[i].to(device) + 1) * 127.5
        mask = torch.ones_like(img2, device=device)
        img2_warp = torch_tps_transform.transformer(
            torch.cat([img2, mask], 1), norm_mesh, norm_rigid_mesh, (img_h, img_w), mode=warp_mode
        )
        stable_list.append(img2_warp[0].detach().cpu().numpy().transpose(1, 2, 0))

    return stable_list


def image_to_tensor(image, normalize=True):
    image = image.astype(np.float32)
    if normalize:
        image = (image / 127.5) - 1.0
    image = np.transpose(image, [2, 0, 1])
    return torch.tensor(image).unsqueeze(0)


def resize_image(image, width, height):
    if image.shape[0] != height or image.shape[1] != width:
        return cv2.resize(image, (width, height))
    return image


def prepare_stage_inputs(ref_sequence, tgt_image, origin_h, origin_w, net_h, net_w, video_length):
    ref_hr_tensor_list = []
    tgt_hr_tensor_list = []
    ref_tensor_list = []
    tgt_tensor_list = []
    ref_metric_list = []

    tgt_origin = resize_image(tgt_image, origin_w, origin_h)
    tgt_low = resize_image(tgt_origin, net_w, net_h)
    tgt_hr_tensor = image_to_tensor(tgt_origin, normalize=True)
    tgt_tensor = image_to_tensor(tgt_low, normalize=True)

    for k in range(video_length):
        ref_origin = resize_image(ref_sequence[k], origin_w, origin_h)
        ref_low = resize_image(ref_origin, net_w, net_h)

        ref_hr_tensor_list.append(image_to_tensor(ref_origin, normalize=True))
        tgt_hr_tensor_list.append(tgt_hr_tensor.clone())
        ref_tensor_list.append(image_to_tensor(ref_low, normalize=True))
        tgt_tensor_list.append(tgt_tensor.clone())
        ref_metric_list.append(ref_low.astype(np.float32))

    return {
        'ref_hr_tensor_list': ref_hr_tensor_list,
        'tgt_hr_tensor_list': tgt_hr_tensor_list,
        'ref_tensor_list': ref_tensor_list,
        'tgt_tensor_list': tgt_tensor_list,
        'ref_metric_list': ref_metric_list,
    }


def load_models(device):
    spatial_net = SpatialNet().to(device)
    temporal_net = TemporalNet().to(device)
    smooth_net = SmoothNet().to(device)

    ckpt_list = glob.glob(os.path.join(MODEL_DIR, '*.pth'))
    ckpt_list.sort()
    if len(ckpt_list) != 3:
        raise FileNotFoundError(f'Expected 3 checkpoints in {MODEL_DIR}, but found {len(ckpt_list)}')

    spatial_model_path = os.path.join(MODEL_DIR, 'spatial_warp.pth')
    temporal_model_path = os.path.join(MODEL_DIR, 'temporal_warp.pth')
    smooth_model_path = os.path.join(MODEL_DIR, 'smooth_warp.pth')

    spatial_checkpoint = torch.load(spatial_model_path, map_location='cpu')
    temporal_checkpoint = torch.load(temporal_model_path, map_location='cpu')
    smooth_checkpoint = torch.load(smooth_model_path, map_location='cpu')

    spatial_net.load_state_dict(spatial_checkpoint['model'])
    temporal_net.load_state_dict(temporal_checkpoint['model'])
    smooth_net.load_state_dict(smooth_checkpoint['model'])
    logger.info(f'load model from {spatial_model_path}')
    logger.info(f'load model from {temporal_model_path}')
    logger.info(f'load model from {smooth_model_path}')

    spatial_net.eval()
    temporal_net.eval()
    smooth_net.eval()
    return spatial_net, temporal_net, smooth_net


def summarize_metric(metric_name, metric_list):
    if not metric_list:
        logger.warning(f'No valid {metric_name} values were collected.')
        return

    sorted_metric = sorted(metric_list, reverse=True)
    total = len(sorted_metric)
    thirty_percent_index = int(total * 0.3)
    sixty_percent_index = int(total * 0.6)

    metric_top_30 = sorted_metric[0:thirty_percent_index]
    metric_top_60 = sorted_metric[thirty_percent_index:sixty_percent_index]
    metric_top_100 = sorted_metric[sixty_percent_index:]

    if metric_top_30:
        logger.info(f'[{metric_name}] top 30%: {np.mean(metric_top_30)}')
    if metric_top_60:
        logger.info(f'[{metric_name}] top 30~60%: {np.mean(metric_top_60)}')
    if metric_top_100:
        logger.info(f'[{metric_name}] top 60~100%: {np.mean(metric_top_100)}')
    logger.info(f'[{metric_name}] average: {np.mean(sorted_metric)}')


def run_stage(
    ref_sequence,
    tgt_image,
    origin_h,
    origin_w,
    device,
    spatial_net,
    temporal_net,
    smooth_net,
    net_h,
    net_w,
    video_length,
    warp_mode,
    fusion_mode,
):
    sample = prepare_stage_inputs(ref_sequence, tgt_image, origin_h, origin_w, net_h, net_w, video_length)
    ref_tensor_list = [tensor.to(device) for tensor in sample['ref_tensor_list']]
    tgt_tensor_list = [tensor.to(device) for tensor in sample['tgt_tensor_list']]

    tmotion_tensor_list = []
    smotion_tensor_list = []
    omask_tensor_list = []

    start_time = time.time()
    for k in range(video_length):
        with torch.no_grad():
            spatial_batch_out = build_SpatialNet(spatial_net, ref_tensor_list[k], tgt_tensor_list[k])
        smotion_tensor_list.append(spatial_batch_out['motion'])
        omask_tensor_list.append(spatial_batch_out['overlap_mesh'])

    with torch.no_grad():
        temporal_batch_out = build_TemporalNet(temporal_net, tgt_tensor_list)
    tmotion_tensor_list = temporal_batch_out['motion_list']
    logger.info(f'fps (spatial & temporal warp): {video_length / max(time.time() - start_time, 1e-6):.4f}')

    rigid_mesh = get_rigid_mesh(1, net_h, net_w, device)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, net_h, net_w)
    smesh_list = []
    tsmotion_list = []
    for k in range(len(tmotion_tensor_list)):
        smotion = smotion_tensor_list[k]
        smesh = rigid_mesh + smotion
        if k == 0:
            tsmotion = smotion.clone() * 0
        else:
            smotion_prev = smotion_tensor_list[k - 1]
            smesh_prev = rigid_mesh + smotion_prev
            tmotion = tmotion_tensor_list[k]
            tmesh = rigid_mesh + tmotion
            norm_smesh_prev = get_norm_mesh(smesh_prev, net_h, net_w)
            norm_tmesh = get_norm_mesh(tmesh, net_h, net_w)
            tsmesh = torch_tps_transform_point.transformer(norm_tmesh, norm_rigid_mesh, norm_smesh_prev)
            tsmotion = recover_mesh(tsmesh, net_h, net_w) - smesh
        smesh_list.append(smesh)
        tsmotion_list.append(tsmotion)

    ori_mesh = None
    target_mesh = None
    for k in range(len(tmotion_tensor_list) - 6):
        tsmotion_sublist = list(tsmotion_list[k:k + 7])
        tsmotion_sublist[0] = smotion_tensor_list[k] * 0

        with torch.no_grad():
            smooth_batch_out = build_SmoothNet(smooth_net, tsmotion_sublist, smesh_list[k:k + 7], omask_tensor_list[k:k + 7])

        current_ori_mesh = smooth_batch_out['ori_mesh']
        current_target_mesh = smooth_batch_out['target_mesh']
        if k == 0:
            ori_mesh = current_ori_mesh
            target_mesh = current_target_mesh
        else:
            ori_mesh = torch.cat((ori_mesh, current_ori_mesh[:, -1, ...].unsqueeze(1)), 1)
            target_mesh = torch.cat((target_mesh, current_target_mesh[:, -1, ...].unsqueeze(1)), 1)

    logger.info(f'fps (smooth warp): {video_length / max(time.time() - start_time, 1e-6):.4f}')

    stable_list, out_width, out_height = get_stable_sqe(
        sample['ref_hr_tensor_list'],
        sample['tgt_hr_tensor_list'],
        target_mesh,
        warp_mode=warp_mode,
        fusion_mode=fusion_mode,
    )
    logger.info(f'fps (warping & blending): {video_length / max(time.time() - start_time, 1e-6):.4f}')

    stable_metric_list = get_stable_sqe_metric(sample['tgt_tensor_list'], target_mesh, warp_mode='NORMAL')
    pair_psnr_list = []
    pair_ssim_list = []
    for k in range(video_length):
        ref_img = sample['ref_metric_list'][k]
        img2_warp = stable_metric_list[k][..., 0:3]
        img2_warp_mask = stable_metric_list[k][..., 3:6]
        pair_psnr_list.append(compare_psnr(ref_img * img2_warp_mask, img2_warp * img2_warp_mask, 255))
        pair_ssim_list.append(compare_ssim(ref_img * img2_warp_mask, img2_warp * img2_warp_mask, 255))

    return {
        'stable_list': stable_list,
        'out_width': out_width,
        'out_height': out_height,
        'pair_psnr_list': pair_psnr_list,
        'pair_ssim_list': pair_ssim_list,
        'pair_psnr_avg': float(np.mean(pair_psnr_list)),
        'pair_ssim_avg': float(np.mean(pair_ssim_list)),
        'ori_mesh': ori_mesh,
        'target_mesh': target_mesh,
    }


def test(args):
    os.environ['CUDA_DEVICES_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not torch.cuda.is_available():
        raise RuntimeError('Current StabStitch sv_comp inference requires CUDA. Please run it on a GPU machine.')
    if args.video_length < 7:
        raise ValueError('--video_length must be at least 7 for the smooth warp buffer.')

    device = torch.device('cuda')
    sv_comp_cfg = OmegaConf.load('sv_comp/sv_comp.yaml')
    with open('sv_comp/intrinsics.yaml', 'r', encoding='utf-8') as file:
        intrinsics = yaml.safe_load(file)
    dataset = MultiWarpDataset(config=sv_comp_cfg, intrinsics=intrinsics, is_train=False)
    if len(dataset) == 0:
        logger.warning('No samples were found under the current sv_comp configuration.')
        return

    spatial_net, temporal_net, smooth_net = load_models(device)
    logger.info('##################start testing#######################')

    sample_psnr_list = []
    sample_ssim_list = []
    net_h = 360
    net_w = 480

    for idx in range(len(dataset)):
        input_imgs, _ = dataset[idx]
        origin_h, origin_w = input_imgs[0].shape[0], input_imgs[0].shape[1]
        sample_path = dataset.get_path(idx)
        result_path = os.path.join(sample_path, args.output_dir_name)
        os.makedirs(result_path, exist_ok=True)

        logger.info(f'---------------{idx}---------------')
        logger.info(sample_path)

        middle_stitch_results = None
        batch_psnr_list = []
        batch_ssim_list = []
        for stage_idx in range(sv_comp_cfg.input_img_num - 1):
            if stage_idx == 0:
                ref_sequence = [input_imgs[stage_idx]] * args.video_length
            else:
                ref_sequence = middle_stitch_results
            tgt_image = input_imgs[stage_idx + 1]

            stage_output = run_stage(
                ref_sequence=ref_sequence,
                tgt_image=tgt_image,
                origin_h=origin_h,
                origin_w=origin_w,
                device=device,
                spatial_net=spatial_net,
                temporal_net=temporal_net,
                smooth_net=smooth_net,
                net_h=net_h,
                net_w=net_w,
                video_length=args.video_length,
                warp_mode=args.warp_mode,
                fusion_mode=args.fusion_mode,
            )

            middle_stitch_results = stage_output['stable_list']
            batch_psnr_list.append(stage_output['pair_psnr_avg'])
            batch_ssim_list.append(stage_output['pair_ssim_avg'])

            output_name = f'{sv_comp_cfg.input_img_num}_{stage_idx + 2}.jpg'
            output_img = np.clip(middle_stitch_results[args.video_length // 2], 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(result_path, output_name), output_img)
            logger.info(f"pair_psnr_list: {stage_output['pair_psnr_list']}, pair_psnr_avg: {stage_output['pair_psnr_avg']}")
            logger.info(f"pair_ssim_list: {stage_output['pair_ssim_list']}, pair_ssim_avg: {stage_output['pair_ssim_avg']}")

        batch_psnr_avg = float(np.mean(batch_psnr_list))
        batch_ssim_avg = float(np.mean(batch_ssim_list))
        logger.info(f'batch_psnr_list: {batch_psnr_list}, batch_psnr_avg: {batch_psnr_avg}')
        logger.info(f'batch_ssim_list: {batch_ssim_list}, batch_ssim_avg: {batch_ssim_avg}')
        sample_psnr_list.append(batch_psnr_avg)
        sample_ssim_list.append(batch_ssim_avg)

    logger.info('<==================== Analysis ===================>')
    logger.info(f'Total num: {len(dataset)}')
    summarize_metric('psnr', sample_psnr_list)
    summarize_metric('ssim', sample_ssim_list)
    logger.info('##################end testing#######################')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--video_length', type=int, default=8)
    parser.add_argument('--output_dir_name', type=str, default='stabstitch')
    parser.add_argument('--warp_mode', type=str, default='FAST')
    parser.add_argument('--fusion_mode', type=str, default='AVERAGE')

    print('<==================== Loading data ===================>\n')
    args = parser.parse_args()
    print(args)
    test(args)
