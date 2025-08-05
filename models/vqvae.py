import torch
import numpy as np 
from loguru import logger
import torch.nn as nn  
from utils.builder import MODELS 
from einops import rearrange
from models import *
from vector_quantize_pytorch import LFQ

def patchify(features,resolution):
    return rearrange(features,'b c (n1 n2 n3) (p1 p2 p3) -> b c (n1 p1) (n2 p2) (n3 p3)',n1=2,n2=2,n3=8,p1=resolution,p2=resolution,p3=resolution)

def unpatchify(features,resolution):
    return rearrange(features,'b c (n1 p1) (n2 p2) (n3 p3) -> b (n1 n2 n3) (p1 p2 p3 c)',n1=2,n2=2,n3=8,p1=resolution,p2=resolution,p3=resolution)


@MODELS.register_module()
class VQVAE(nn.Module):
    def __init__(self, config):
        super(VQVAE, self).__init__()
        self.resolution = config.resolution
        self.local_encoders = nn.ModuleList([PointNetEncoder(config) for _ in range(32)])
        if self.resolution==3:
            self.unet_encoder = UNet3D(config.zdim,config.zdim,num_levels=3)
        else:
            self.unet_encoder = UNet3D(config.zdim,config.zdim,num_levels=4)
        self.npoint = config.npoint
        self.quantizer = LFQ(
            codebook_size = 4096,      
            dim = config.zdim,                   
            entropy_loss_weight = 1.,  
            diversity_gamma = 1.,        
            experimental_softplus_entropy_loss=True
        )
        if self.resolution == 3:
            self.unet_decoder = UNet3D(config.zdim,config.zdim,num_levels=3)
        else:
            self.unet_decoder = UNet3D(config.zdim,config.zdim,num_levels=4)
        self.final_dim = config.zdim * (self.resolution ** 3)
        self.mlps = nn.ModuleList([nn.Linear(self.final_dim,self.npoint*3) for _ in range(32)])
        self.mask_predictor = nn.ModuleList([nn.Linear(self.final_dim,1) for _ in range(32)])

    def forward(self, points):
        """
        points: [B, N, P, 3]
        """
        features = torch.stack([self.local_encoders[i].encode(points[:,i]) for i in range(32)],dim=2) 
        features = patchify(features,self.resolution)
        z = self.unet_encoder(features)
        z, indices, entropy_aux_loss = self.quantizer(z) 
        decoded = unpatchify(self.unet_decoder(z),self.resolution)
        rec = torch.stack([self.mlps[i](decoded[:,i]) for i in range(32)],dim=1)
        masks = torch.stack([self.mask_predictor[i](decoded[:,i]) for i in range(32)],dim=1).squeeze(2)
        rec = rearrange(rec,'b n (p c)-> b n p c',c=3)
        return rec, entropy_aux_loss, masks
    
    def encode(self, points):
        features = torch.stack([self.local_encoders[i].encode(points[:,i]) for i in range(32)],dim=2)  # [B,C,N,P]
        features = patchify(features,self.resolution)
        z = self.unet_encoder(features)
        return z

    def decode(self,z):
        z = rearrange(z,'b c w h d -> b (w h d) c')
        z, indices, entropy_aux_loss = self.quantizer(z) 
        z = rearrange(z,'b (w h d) c -> b c w h d',w=2*self.resolution,h=2*self.resolution,d=8*self.resolution)
        decoded = unpatchify(self.unet_decoder(z),self.resolution)
        rec = torch.stack([self.mlps[i](decoded[:,i]) for i in range(32)],dim=1)
        masks = torch.stack([self.mask_predictor[i](decoded[:,i]) for i in range(32)],dim=1).squeeze(2)
        rec = rearrange(rec,'b n (p c)-> b n p c',c=3)
        return rec, masks


