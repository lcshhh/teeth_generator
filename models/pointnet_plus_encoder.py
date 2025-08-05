# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import torch.nn as nn 
from loguru import logger 
from models.pvcnn2 import create_pointnet2_sa_components 
from utils.builder import MODELS 
from models import *
from timm.models.layers import trunc_normal_
import torch

@MODELS.register_module()
class PointNetEncoder(nn.Module):
    force_att = 0
    def __init__(self, config):       
        super().__init__()
        self.sa_blocks = [
        [[32, 2, 32], [1024, 0.1, 32, [32, 32]]],
        [[32, 1, 16], [256, 0.2, 32, [32, 64]]],
        [[32, 1, 8], [config.resolution**3, 0.4, 32, [64, 128]]]
        ]
        sa_blocks = self.sa_blocks 
        layers, sa_in_channels, channels_sa_features, _  = \
            create_pointnet2_sa_components(sa_blocks, 
            extra_feature_channels=0, input_dim=config.input_dim, 
            embed_dim=0, force_att=self.force_att,
            use_att=True, with_se=True)
        self.zdim = config.zdim 
        self.layers = nn.ModuleList(layers) 
        self.voxel_dim = [n[1][-1][-1] for n in self.sa_blocks]

    def forward(self, x):
        x = x.transpose(1, 2) # B,3,N
        xyz = x ## x[:,:3,:]
        features = x
        for layer_id, layer in enumerate(self.layers):
            features, xyz, _ = layer( (features, xyz, None) )
        # features: B,D,N
        return features
    
    def encode(self,x):
        x = x.transpose(1, 2) # B,3,N
        xyz = x ## x[:,:3,:]
        features = x
        for layer_id, layer in enumerate(self.layers):
            features, xyz, _ = layer( (features, xyz, None) )
        # features: B,D,N
        return features
    

class StyleEncoder(nn.Module):
    force_att = 0 # add attention to all layers  
    def __init__(self, config):
        super().__init__()
        sa_blocks = [
        [[32, 2, 32], [1024, 0.1, 32, [32, 32]]],
        [[32, 1, 16], [256, 0.2, 32, [32, 64]]],
        [[32, 1, 8], [128, 0.4, 32, [64, config.zdim]]]
        ]
        self.sa_blocks = sa_blocks
        layers, sa_in_channels, channels_sa_features, _  = \
            create_pointnet2_sa_components(sa_blocks, 
            extra_feature_channels=0, input_dim=config.input_dim, 
            embed_dim=0, force_att=self.force_att,
            use_att=True, with_se=True)
        self.mlp_mean = nn.Linear(128,config.zdim) 
        self.mlp_var = nn.Linear(128,config.zdim)
        self.pe = nn.Parameter(torch.zeros(config.zdim))
        # self.ln = nn.LayerNorm(config.zdim)
        trunc_normal_(self.pe,std=0.02)
        self.layers = nn.ModuleList(layers) 
        self.voxel_dim = [n[1][-1][-1] for n in self.sa_blocks]
    
    def forward(self,x):
        x = x.transpose(1, 2) # B,3,N
        xyz = x ## x[:,:3,:]
        features = x
        for layer_id, layer in enumerate(self.layers):
            features, xyz, _ = layer( (features, xyz, None) )
        # features: B,D,N; xyz: B,3,N
        style_mean = torch.mean(features,1)
        style_var = torch.var(features,1)
        style_feature = self.mlp_mean(style_mean) + self.mlp_var(style_var) + self.pe.expand(x.shape[0],-1)
        return style_feature

class ShapeEncoder(nn.Module):
    sa_blocks = [
        [[32, 2, 32], [1024, 0.1, 32, [32, 32]]],
        [[32, 1, 16], [256, 0.2, 32, [32, 128]]]
        ]
    force_att = 0 # add attention to all layers  
    def __init__(self, config):
        super().__init__()
        sa_blocks = self.sa_blocks 
        layers, sa_in_channels, channels_sa_features, _  = \
            create_pointnet2_sa_components(sa_blocks, 
            extra_feature_channels=0, input_dim=config.input_dim, 
            embed_dim=0, force_att=self.force_att,
            use_att=True, with_se=True)
        self.mlp = nn.Linear(channels_sa_features, config.zdim) 
        self.zdim = config.zdim 
        self.layers = nn.ModuleList(layers) 
        self.voxel_dim = [n[1][-1][-1] for n in self.sa_blocks]

    def forward(self, x):
        """
        Args: 
            x: B,N,3 
        Returns: 
            mu, sigma: B,D
        """
        # output = {} 
        x = x.transpose(1, 2) # B,3,N
        xyz = x ## x[:,:3,:]
        features = x
        for layer_id, layer in enumerate(self.layers):
            features, xyz, _ = layer( (features, xyz, None) )
        # features: B,D,N; xyz: B,3,N
       
        features = features.max(-1)[0]
        features = self.mlp(features)

        return features

