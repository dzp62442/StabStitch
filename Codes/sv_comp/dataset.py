from torch.utils.data import Dataset
import  numpy as np
import cv2, torch
import os, re
import glob
from collections import OrderedDict
import random
from loguru import logger


class MultiWarpDataset(Dataset):

    def __init__(self, config: dict, intrinsics: dict, is_train=False):
        self.net_input_width = config['net_input_width']
        self.net_input_height = config['net_input_height']
        self.data_path = config['data_path']
        self.sub_datasets = config['sub_datasets']
        self.input_img_num = config['input_img_num']
        self.mode = 'training' if is_train else 'testing'
        self.intrinsics = intrinsics
        self.datas = OrderedDict()
        self.xmaps, self.ymaps = OrderedDict(), OrderedDict()  # 投影映射矩阵
        self.input_masks = OrderedDict()  # 输入图像的掩码
        
        # 加载数据集
        for sub_dataset in self.sub_datasets:  # 遍历所有的子数据集
            groups = glob.glob(os.path.join(self.data_path, sub_dataset, self.mode, '*'))  # 当前子数据集中的所有数据组
            pattern = r'^\d{4,5}$'  # 匹配仅由四位或五位数字组成的字符串
            for group in sorted(groups):
                group_idx = group.split('/')[-1]
                if bool(re.match(pattern, group_idx)):  # 如果当前数据组的名称符合要求
                    img_lists = glob.glob(os.path.join(group, '*.jpg'))
                    if len(img_lists) < self.input_img_num:
                        continue
                    data_key = '/'.join(group.split('/')[-3:])  # 格式为 sub_dataset/mode/group_idx
                    self.datas[data_key] = {}
                    self.datas[data_key]['path'] = group
                    self.datas[data_key]['image'] = img_lists
                    self.datas[data_key]['image'].sort()
        self.data_keys = list(self.datas.keys())
        logger.info(self.sub_datasets)

        # 初始化投影映射矩阵
        self.use_warp = config['use_warp']
        if (self.use_warp):
            self.warp_mode = config['warp_cfg']['warp_mode']
            for sub_dataset in self.sub_datasets:  # 遍历所有的子数据集
                sub_dataset_keys = [data_key for data_key in self.data_keys if sub_dataset in data_key]  # self.data_keys 中包含的当前子数据集的所有数据
                init_img = cv2.imread(self.datas[sub_dataset_keys[0]]['image'][0])
                if sub_dataset == 'UDIS':  # UDIS数据集不进行投影变换
                    self.xmaps[sub_dataset], self.ymaps[sub_dataset],  = None, None
                    self.input_masks[sub_dataset] = np.ones((init_img.shape[0], init_img.shape[1]), np.uint8)  # 生成与图像大小相同的掩码
                elif sub_dataset == 'PandaSet':  # PandaSet 数据集包含多种相机内参
                    self.xmaps[sub_dataset], self.ymaps[sub_dataset], self.input_masks[sub_dataset] = OrderedDict(), OrderedDict(), OrderedDict()
                    K_50 = np.array(self.intrinsics['PandaSet_50']).astype(np.float32)
                    xmap_50, ymap_50, mask_50 = self.init_warp_map_cv(init_img, K_50, config['warp_cfg']['scale'])
                    self.xmaps[sub_dataset]['FF'], self.ymaps[sub_dataset]['FF'], self.input_masks[sub_dataset]['FF'] = xmap_50, ymap_50, mask_50
                    K_107 = np.array(self.intrinsics['PandaSet_107']).astype(np.float32)
                    xmap_107, ymap_107, mask_107 = self.init_warp_map_cv(init_img, K_107, config['warp_cfg']['scale'])
                    self.xmaps[sub_dataset]['LB'], self.ymaps[sub_dataset]['LB'], self.input_masks[sub_dataset]['LB'] = xmap_107, ymap_107, mask_107
                    self.xmaps[sub_dataset]['LF'], self.ymaps[sub_dataset]['LF'], self.input_masks[sub_dataset]['LF'] = xmap_107, ymap_107, mask_107
                    self.xmaps[sub_dataset]['RF'], self.ymaps[sub_dataset]['RF'], self.input_masks[sub_dataset]['RF'] = xmap_107, ymap_107, mask_107
                    self.xmaps[sub_dataset]['RB'], self.ymaps[sub_dataset]['RB'], self.input_masks[sub_dataset]['RB'] = xmap_107, ymap_107, mask_107
                else:  # 其他数据集只包含一种相机内参
                    if (sub_dataset == 'PowerFlowTractor1' or sub_dataset == 'MiniTank1'):
                        K = np.array(self.intrinsics['Leopard_94']).astype(np.float32)
                    elif (sub_dataset == 'WhiteCar' or sub_dataset == 'RealTractor3' or sub_dataset == 'RealTractor5' or sub_dataset == 'RealTractor6'):
                        K = np.array(self.intrinsics['Sensing_120']).astype(np.float32)
                    elif (sub_dataset == 'NuScenes'):  # 目前 NuScenes 数据集仅使用了视场角均为 70° 的相机图像
                        K = np.array(self.intrinsics['NuScenes_70']).astype(np.float32)
                    xmap, ymap, mask = self.init_warp_map_cv(init_img, K, config['warp_cfg']['scale'])
                    self.xmaps[sub_dataset], self.ymaps[sub_dataset], self.input_masks[sub_dataset] = xmap, ymap, mask


    def __getitem__(self, index):
        input_index = [x for x in range(self.input_img_num)]  # 读取的所有图像的序号，[0, self.input_img_num - 1]

        input_imgs, input_masks = [], []
        for i in range(self.input_img_num):
            img_path = self.datas[self.data_keys[index]]['image'][input_index[i]]
            img_name = img_path.split('/')[-1].split('.')[0]
            input_img = cv2.imread(img_path)
            input_mask = None
            # 对图像进行投影变换
            if (self.use_warp):
                sub_dataset = self.data_keys[index].split('/')[0]  # 当前数据所属的子数据集
                if sub_dataset == 'UDIS':  # UDIS数据集不进行投影变换
                    input_mask = self.input_masks[sub_dataset]
                elif sub_dataset == 'PandaSet':  # PandaSet 数据集包含多种相机内参
                    img_suffix = img_name.split('_')[-1]  # 图像名称后缀，用于区分不同相机
                    input_img = cv2.remap(input_img, self.xmaps[sub_dataset][img_suffix], self.ymaps[sub_dataset][img_suffix], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
                    input_mask = self.input_masks[sub_dataset][img_suffix]
                else:  # 其他数据集只包含一种相机内参
                    input_img = cv2.remap(input_img, self.xmaps[sub_dataset], self.ymaps[sub_dataset], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
                    input_mask = self.input_masks[sub_dataset]
            else:
                input_mask = np.ones((input_img.shape[0], input_img.shape[1]), np.uint8)  # 生成与图像大小相同的掩码
            # 将图像缩放到网络输入尺寸
            # if (self.net_input_height != input_img.shape[0] or self.net_input_width != input_img.shape[1]):
            #     input_img = cv2.resize(input_img, (self.net_input_width, self.net_input_height))
            #     input_mask = cv2.resize(input_mask, (self.net_input_width, self.net_input_height))
            # 直接返回 cv 格式
            input_imgs.append(input_img)
            input_masks.append(input_mask)

        return (input_imgs, input_masks)
            
    def to_tensor(self, input_img, input_mask=None):
        # 将输入图像转化为 tensor
        input_img = input_img.astype(dtype=np.float32)
        input_img = (input_img / 127.5) - 1.0  # 像素值从 0~255 转换到 -1~1
        input_img = np.transpose(input_img, [2, 0, 1])  # 将图像从 (H, W, C) 转换为 (C, H, W)
        input_tensor = torch.tensor(input_img)
        input_tensor = input_tensor.unsqueeze(0)  # 不使用 dataloader，手动添加 bs 维度
        # 将输入掩码转化为 tensor
        if input_mask is not None:
            input_mask = np.expand_dims(input_mask, axis=2)  # 添加通道维度
            input_mask = np.repeat(input_mask, 3, axis=2)  # 单通道扩展为三通道
            input_mask = input_mask.astype(dtype=np.float32)
            input_mask = np.transpose(input_mask, [2, 0, 1])
            input_mask_tensor = torch.tensor(input_mask)
            input_mask_tensor = input_mask_tensor.unsqueeze(0)  # 不使用 dataloader，手动添加 bs 维度
            return input_tensor, input_mask_tensor
        else:
            return input_tensor

    def __len__(self):
        return len(self.datas.keys())
    
    def get_path(self, index):
        return self.datas[self.data_keys[index]]['path']
    
    # 初始化圆柱面投影的映射矩阵，使用opencv的PyRotationWarper
    def init_warp_map_cv(self, init_img, K, scale):
        warper = cv2.PyRotationWarper(self.warp_mode, scale)  # TODO: scale的具体含义？
        src_size = (init_img.shape[1], init_img.shape[0])  # (width, height)
        R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).astype(np.float32)
        retval, xmap, ymap = warper.buildMaps(src_size, K, R)  # 生成投影映射矩阵
        mask = np.ones((init_img.shape[0], init_img.shape[1]), np.uint8)  # 生成与图像大小相同的掩码
        mask_remaped = cv2.remap(mask, xmap, ymap, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)  # 使用映射矩阵对掩码进行投影变换
        return xmap, ymap, mask_remaped
