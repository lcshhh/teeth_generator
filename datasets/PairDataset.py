
import json
import random
from pathlib import Path
import numpy as np
import os
import torch
import torch.utils.data as data
import trimesh
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import matrix_to_rotation_6d
import copy
from utils.builder import DATASETS
import open3d as o3d
from utils.pointcloud import read_pointcloud

@DATASETS.register_module()
class PairDataset(data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.after_path = config.after_path
        self.before_path = config.before_path
        self.npoint = config.npoint
        with open(config.file) as f:
            self.indexes = [int(i.strip()) for i in f.readlines()]
        self.train = config.train
    
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
            before_path = os.path.join(self.before_path,f'{index}_{i}.ply')
            if os.path.exists(after_path) and os.path.exists(before_path):
                masks[i] = 1
                after = read_pointcloud(after_path)
                after_points[i] = after[:point_num]
                after_centroid[i] = after[point_num] 
                before = read_pointcloud(before_path)
                before_points[i] = before[:point_num]
                before_centroid[i] = before[point_num]
        return   index,before_points,before_centroid,after_points,after_centroid,masks


    def __len__(self):
        return len(self.indexes)