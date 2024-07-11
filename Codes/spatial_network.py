import torch
import torch.nn as nn
import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
import utils.torch_tps_transform as torch_tps_transform
import ssl
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.models as models



grid_h = 6
grid_w = 8


#Covert global homo into mesh
def H2Mesh(H, rigid_mesh):

    H_inv = torch.inverse(H)
    ori_pt = rigid_mesh.reshape(rigid_mesh.size()[0], -1, 2)
    ones = torch.ones(rigid_mesh.size()[0], (grid_h+1)*(grid_w+1),1)
    if torch.cuda.is_available():
        ori_pt = ori_pt.cuda()
        ones = ones.cuda()

    ori_pt = torch.cat((ori_pt, ones), 2) # bs*(grid_h+1)*(grid_w+1)*3
    tar_pt = torch.matmul(H_inv, ori_pt.permute(0,2,1)) # bs*3*(grid_h+1)*(grid_w+1)

    mesh_x = torch.unsqueeze(tar_pt[:,0,:]/tar_pt[:,2,:], 2)
    mesh_y = torch.unsqueeze(tar_pt[:,1,:]/tar_pt[:,2,:], 2)
    mesh = torch.cat((mesh_x, mesh_y), 2).reshape([rigid_mesh.size()[0], grid_h+1, grid_w+1, 2])

    return mesh

# get rigid mesh
def get_rigid_mesh(batch_size, height, width):

    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    if torch.cuda.is_available():
        ww = ww.cuda()
        hh = hh.cuda()

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2) # (grid_h+1)*(grid_w+1)*2
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt

# normalize mesh from -1 ~ 1
def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = torch.stack([mesh_w, mesh_h], 3) # bs*(grid_h+1)*(grid_w+1)*2

    return norm_mesh.reshape([batch_size, -1, 2]) # bs*-1*2




def build_SpatialNet(net, input1_tensor, input2_tensor):
    batch_size, _, img_h, img_w = input1_tensor.size()

    # input1_tensor = (input1_tensor / 127.5) - 1.0
    # input2_tensor = (input2_tensor / 127.5) - 1.0

    H_motion, mesh_motion = net(input1_tensor, input2_tensor)

    H_motion = H_motion.reshape(-1, 4, 2)
    mesh_motion = mesh_motion.reshape(-1, grid_h+1, grid_w+1, 2)

    # initialize the source points bs x 4 x 2
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
    if torch.cuda.is_available():
        src_p = src_p.cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    # target points
    dst_p = src_p + H_motion
    # solve homo using DLT
    H = torch_DLT.tensor_DLT(src_p, dst_p)

    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = H2Mesh(H, rigid_mesh)
    mesh = ini_mesh + mesh_motion

    # calculate the overlapping regions to apply shape-preserving constraints
    overlap_one = torch.ones_like(mesh[...,0])
    overlap_zero = torch.zeros_like(mesh[...,0])

    overlap_w = torch.where((mesh[...,0]<img_w) & (mesh[...,0]>=0), overlap_one, overlap_zero)
    overlap_h = torch.where((mesh[...,1]<img_h) & (mesh[...,1]>=0), overlap_one, overlap_zero)
    overlap_mesh = overlap_w * overlap_h
    overlap_mesh = torch.clamp(overlap_mesh.unsqueeze(3), 0, 1)

    out_dict = {}
    out_dict.update(motion = mesh - rigid_mesh, overlap_mesh = overlap_mesh)


    return out_dict




def get_res18_FeatureMap(resnet18_model):
    # feature extractor
    # suppose input resolution to be 256*256   |   512*384
    # the output resolution should be 16*16    |    32*24

    layers_list = []

    # mask
    #layers_list.append(nn.Conv2d(6, 3, kernel_size=3, padding=1, bias=False))

    layers_list.append(resnet18_model.conv1)    #stride 2*2     H/2
    #layers_list.append(nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
    layers_list.append(resnet18_model.bn1)
    layers_list.append(resnet18_model.relu)
    layers_list.append(resnet18_model.maxpool)  #stride 2       H/4

    layers_list.append(resnet18_model.layer1)                  #H/4
    layers_list.append(resnet18_model.layer2)                  #H/8

    feature_extractor_stage1 = nn.Sequential(*layers_list)
    feature_extractor_stage2 = nn.Sequential(resnet18_model.layer3)

    # for img of 360*640
    # the resolution of feature1 should be 45*80
    # the resolution of feature1 should be *40

    return feature_extractor_stage1, feature_extractor_stage2

# define and forward
class SpatialNet(nn.Module):

    def __init__(self):
        super(SpatialNet, self).__init__()

        self.regressNet1_part1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 12, 20

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 6, 10

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
            # 3, 5
        )

        self.regressNet1_part2 = nn.Sequential(
            nn.Linear(in_features=768, out_features=512, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=128, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=8, bias=True)
        )


        self.regressNet2_part1 = nn.Sequential(
            nn.Conv2d(121, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 23, 40

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 12, 20

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 6, 10

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
            # 3, 5
        )

        self.regressNet2_part2 = nn.Sequential(
            nn.Linear(in_features=1536, out_features=1024, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=512, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=(grid_w+1)*(grid_h+1)*2, bias=True)

        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        resnet18_model = models.resnet.resnet18(weights="DEFAULT")
        if torch.cuda.is_available():
            resnet18_model = resnet18_model.cuda()
        self.feature_extractor_stage1, self.feature_extractor_stage2 = get_res18_FeatureMap(resnet18_model)
        #-----------------------------------------

    # forward
    def forward(self, input1_tesnor, input2_tesnor):
        batch_size, _, img_h, img_w = input1_tesnor.size()

        # print("111")
        feature_1_64 = self.feature_extractor_stage1(input1_tesnor)
        feature_1_32 = self.feature_extractor_stage2(feature_1_64)
        feature_2_64 = self.feature_extractor_stage1(input2_tesnor)
        feature_2_32 = self.feature_extractor_stage2(feature_2_64)
        # print("222")
        ######### stage 1
        correlation_32 = self.CCL(feature_1_32, feature_2_32)
        temp_1 = self.regressNet1_part1(correlation_32)
        temp_1 = temp_1.view(temp_1.size()[0], -1)
        offset_1 = self.regressNet1_part2(temp_1)
        H_motion_1 = offset_1.reshape(-1, 4, 2)

        # print("333")
        src_p = torch.tensor([[0., 0.], [img_w-1, 0.], [0., img_h-1], [img_w-1, img_h-1]])
        if torch.cuda.is_available():
            src_p = src_p.cuda()
        src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
        dst_p = src_p + H_motion_1
        H = torch_DLT.tensor_DLT(src_p/8, dst_p/8)

        M_tensor = torch.tensor([[img_w/8 / 2.0, 0., img_w/8 / 2.0],
                      [0., img_h/8 / 2.0, img_h/8 / 2.0],
                      [0., 0., 1.]])

        if torch.cuda.is_available():
            M_tensor = M_tensor.cuda()

        M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        M_tensor_inv = torch.inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)
        H_mat = torch.matmul(torch.matmul(M_tile_inv, H), M_tile)

        warp_feature_2_64 = torch_homo_transform.transformer(feature_2_64, H_mat, (int(img_h/8), int(img_w/8)))
        # print("444")
        ######### stage 2
        # correlation_64 = self.CCL(feature_1_64, warp_feature_2_64)
        correlation_64 = self.cost_volume(feature_1_64, warp_feature_2_64, search_range=5, norm=False) # bs, 9^2, 45, 80
        temp_2 = self.regressNet2_part1(correlation_64)
        temp_2 = temp_2.view(temp_2.size()[0], -1)
        offset_2 = self.regressNet2_part2(temp_2)
        # print("555")

        return offset_1, offset_2

    @staticmethod
    def cost_volume(x1, x2, search_range, norm=True, fast=True):
        if norm:
            x1 = F.normalize(x1, p=2, dim=1)
            x2 = F.normalize(x2, p=2, dim=1)
        bs, c, h, w = x1.shape
        padded_x2 = F.pad(x2, [search_range] * 4)  # [b,c,h,w] -> [b,c,h+sr*2,w+sr*2]
        max_offset = search_range * 2 + 1

        if fast:
            # faster(*2) but cost higher(*n) GPU memory
            patches = F.unfold(padded_x2, (max_offset, max_offset)).reshape(bs, c, max_offset ** 2, h, w)
            cost_vol = (x1.unsqueeze(2) * patches).mean(dim=1, keepdim=False)
        else:
            # slower but save memory
            cost_vol = []
            for j in range(0, max_offset):
                for i in range(0, max_offset):
                    x2_slice = padded_x2[:, :, j:j + h, i:i + w]
                    cost = torch.mean(x1 * x2_slice, dim=1, keepdim=True)
                    cost_vol.append(cost)
            cost_vol = torch.cat(cost_vol, dim=1)

        cost_vol = F.leaky_relu(cost_vol, 0.1)

        return cost_vol


    def extract_patches(self, x, kernel=3, stride=1):
        if kernel != 1:
            x = nn.ZeroPad2d(1)(x)
        x = x.permute(0, 2, 3, 1)
        all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
        return all_patches


    def CCL(self, feature_1, feature_2):
        bs, c, h, w = feature_1.size()

        norm_feature_1 = F.normalize(feature_1, p=2, dim=1)
        norm_feature_2 = F.normalize(feature_2, p=2, dim=1)
        #print(norm_feature_2.size())

        patches = self.extract_patches(norm_feature_2)
        if torch.cuda.is_available():
            patches = patches.cuda()

        matching_filters  = patches.reshape((patches.size()[0], -1, patches.size()[3], patches.size()[4], patches.size()[5]))

        match_vol = []
        for i in range(bs):
            single_match = F.conv2d(norm_feature_1[i].unsqueeze(0), matching_filters[i], padding=1)
            match_vol.append(single_match)

        match_vol = torch.cat(match_vol, 0)
        #print(match_vol .size())

        # scale softmax
        softmax_scale = 10
        match_vol = F.softmax(match_vol*softmax_scale,1)

        channel = match_vol.size()[1]

        h_one = torch.linspace(0, h-1, h)
        one1w = torch.ones(1, w)
        if torch.cuda.is_available():
            h_one = h_one.cuda()
            one1w = one1w.cuda()
        h_one = torch.matmul(h_one.unsqueeze(1), one1w)
        h_one = h_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)

        w_one = torch.linspace(0, w-1, w)
        oneh1 = torch.ones(h, 1)
        if torch.cuda.is_available():
            w_one = w_one.cuda()
            oneh1 = oneh1.cuda()
        w_one = torch.matmul(oneh1, w_one.unsqueeze(0))
        w_one = w_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)

        c_one = torch.linspace(0, channel-1, channel)
        if torch.cuda.is_available():
            c_one = c_one.cuda()
        c_one = c_one.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(bs, -1, h, w)

        flow_h = match_vol*(c_one//w - h_one)
        flow_h = torch.sum(flow_h, dim=1, keepdim=True)
        flow_w = match_vol*(c_one%w - w_one)
        flow_w = torch.sum(flow_w, dim=1, keepdim=True)

        feature_flow = torch.cat([flow_w, flow_h], 1)

        return feature_flow
