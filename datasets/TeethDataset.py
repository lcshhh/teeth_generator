
import json
import random
from pathlib import Path
import numpy as np
import os
import torch
import torch.utils.data as data
import trimesh
from scipy.spatial.transform import Rotation
import copy
from utils.builder import DATASETS
import open3d as o3d

def read_pointcloud(path):
    pcd = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pcd.points)
    return xyz

@DATASETS.register_module()
class TeethDataset(data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.after_path = config.after_path
        self.npoint = config.npoint
        with open(config.file) as f:
            self.indexes = [int(i.strip()) for i in f.readlines()]
        self.train = config.train
    
    def __getitem__(self, idx):
        point_num = self.npoint         
        # using np.zeros here will cause PVCNN + PointNet++ to return NaN values.
        after_points = np.random.random((32,point_num,3))/1000
        after_centroid = np.zeros((32,3))
        index = self.indexes[idx]
        masks = np.zeros((32),dtype=np.int32)
        for i in range(32):
            after_path = os.path.join(self.after_path,f'{index}_{i}.ply')
            if os.path.exists(after_path):
                masks[i] = 1
                after = read_pointcloud(after_path)
                after_points[i] = after[:point_num]
                after_centroid[i] = after[point_num] 
        return   index,after_points,after_centroid,masks


    def __len__(self):
        return len(self.indexes)