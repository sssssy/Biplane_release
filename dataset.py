import os
import re

import torch
import torch.utils.data as data
import numpy as np
from prefetch_generator import BackgroundGenerator

import exr
from utils import print, xy_to_xyz, wiwo_xyz_to_hd_thetaphi, thetaphi_to_xyz
from config import RepConfig

class DataLoaderX(data.DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class TrainDataset_allrandom(data.Dataset):

    def __init__(self, config: RepConfig, test_run: bool=False):
        super().__init__()
        self.config = config
        self.dataset_name = type(self).__name__
        self.dataloader = DataLoaderX(
        # self.dataloader = data.DataLoader(
            self,
            batch_size=config.batch_size,
            pin_memory=True,
            num_workers=config.num_workers,
            shuffle=True if not test_run else False,
            drop_last=False
        )
        self.test_run = test_run

    def find_data_root(self, material_name):
        for key in self.config.data_root:
            if material_name.startswith(key+'_'):
                return material_name.replace(key+'_', ''), self.config.data_root[key]
        return material_name, self.config.data_root['default']
            
    def locate(self, num_query, array):
        num = num_query
        for i in range(len(array)):
            if num < array[i]:
                return i, num
            num -= array[i]
        raise ValueError(f'[{self.dataset_name}] locate out of boundary.')

    def __getitem__(self, index):

        material_index, file_index = index // self.config.cache_file_shape[0], index % self.config.cache_file_shape[0]
        material = self.config.train_materials[material_index]
        material, data_dir = self.find_data_root(material)
        
        f = exr.read(os.path.join(
                    os.path.join(self.config.root, data_dir),
                    material, 
                    f'{self.config.cache_file_shape[1]}x{self.config.cache_file_shape[2]}', 
                    f'0_{file_index}.exr'
                ),
                channels=['0_wix', '1_wiy', '2_wox', '3_woy', '4_u', '5_v', '6_R', '7_G', '8_B']
            ).reshape(-1, 9) ## [400, 400, 9] -> [-1, 9]
        
        if self.config.greyscale:
            f[..., -3:] = f[..., -3:].mean(-1, keepdims=True)

        # return material_index, f
                        
        if self.config.random_drop_queries:
            randperm = torch.randperm(f.shape[0])[:f.shape[0] // self.config.random_drop_queries]
            f = f[randperm]
        material_index = np.array(material_index).repeat(f.shape[0])
        view, light, u, v, color = f[..., 0:2], f[..., 2:4], f[..., 4], f[..., 5], f[..., 6:9]
        
        ## process angles 
        view, light = map(torch.from_numpy, [view, light])
        wi, wo = map(xy_to_xyz, [view, light])
        cos = wo[..., -1:]
        h, d = wiwo_xyz_to_hd_thetaphi(wi, wo) ## [bs, 2,], [bs, 2]. phis are in (-pi, pi)
        h, d = thetaphi_to_xyz(h)[..., :2], thetaphi_to_xyz(d)[..., :2] ## [bs, 2], [bs, 2], [-1, 1]x[-1, 1]
        return material_index, wi, cos, h, d, u, v, color

    def __len__(self):
        return self.config.cache_file_shape[0] * len(self.config.train_materials)
