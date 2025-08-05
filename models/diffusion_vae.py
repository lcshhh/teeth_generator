import torch
import torch.nn as nn
from einops import rearrange
from models.vqvae import VQVAE
from models.diffusion import diffusion
from timm.models.vision_transformer import PatchEmbed, Block
from utils.builder import MODELS 
# from util import SharedMLP, LinearMLP


@MODELS.register_module()
class DiffusionVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vae = VQVAE(config)
        if config.args.vae_checkpoint!='':
            checkpoints = torch.load(config.args.vae_checkpoint)
            self.vae.load_state_dict(checkpoints['base_model'])
        self.resolution = config.resolution
        self.zdim = config.zdim
        self.vae.requires_grad_(False)
        self.dpm = diffusion()
    
    def decode(self, z):
        B = z.shape[0]
        out = self.vae.decode(z)
        return out


    def forward(self, points):
        latents = self.vae.encode(points)
        predicted_latents = self.dpm(latents)
        return latents,  predicted_latents
    
    def predict(self, before_pointcloud, before_axis):
        conditions = self.condictor.encode(before_pointcloud, before_axis)
        predicted_latents = self.dpm(None,conditions)
        out = self.decode(predicted_latents)
        return out
    
    def sample(self,sample_num):
        z = torch.randn(sample_num,self.zdim,2*self.resolution,2*self.resolution,8*self.resolution)
        predicted_latents = self.dpm.ddim_sample(z).float()
        points, masks = self.vae.decode(predicted_latents)
        return points, masks
    



