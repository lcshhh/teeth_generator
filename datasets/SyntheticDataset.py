
import json
import random
from pathlib import Path
import numpy as np
import os
import torch
import torch.utils.data as data
import trimesh
from pytorch3d.transforms import matrix_to_rotation_6d
from utils.pointcloud import read_pointcloud
from utils.builder import DATASETS
import open3d as o3d
# You can modify the index of style model here.
style_set = [11]

@DATASETS.register_module()
class SyntheticDataset(data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.after_path = config.after_path
        self.style_path = config.style_path
        self.npoint = config.npoint
        self.style_set = style_set
        with open(config.file) as f:
            self.indexes = [int(i.strip()) for i in f.readlines()]
        
    
    def __getitem__(self, idx):
        point_num = self.npoint
        before_points = np.random.random((32,point_num,3))/1000
        after_points = np.random.random((32,point_num,3))/1000
        before_centroid = np.zeros((32,3))
        after_centroid = np.zeros((32,3))
        index = self.indexes[idx]
        masks = np.zeros((32),dtype=np.int32)
        for i in range(32):
            after_path = os.path.join(self.after_path,f'{index}_{i}.ply')
            if os.path.exists(after_path):
                masks[i] = 1
                after = read_pointcloud(after_path)
                after_points[i] = after[:point_num] 
                after_centroid[i] = np.mean(after_points[i],axis=0)
        rand_idx = random.choice(self.style_set)
        for i in range(32):
            style_path = os.path.join(self.style_path,f'{rand_idx}_{i}.ply')    
            if os.path.exists(style_path):
                before = read_pointcloud(style_path)
                before_points[i] = before[:point_num]
                before_centroid[i] = before[point_num]
        return   index,before_points,before_centroid,after_points,after_centroid,masks


    def __len__(self):
        return len(self.indexes)