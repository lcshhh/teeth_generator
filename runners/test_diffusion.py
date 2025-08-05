import torch
import torch.nn as nn
from utils import builder
import time
from utils.AverageMeter import AverageMeter
from pytorch3d.loss import chamfer_distance
from utils.pointcloud import write_pointcloud
import numpy as np
import vedo
import trimesh
import os
from einops import rearrange
from pytorch3d.transforms import matrix_to_quaternion, Transform3d
from torchvision import transforms
import torch.nn.functional as F

def sample_diffusion(args, config, logger):
    # build dataset
    config.model.args = args
    # build model
    base_model = builder.model_builder(config.model)
    

    # resume ckpts
    # base_model.load_model_from_ckpt(args.ckpts)
    builder.load_model_from_ckpt(base_model,args.ckpts)
 
    logger.info('Using Data parallel ...' , logger = logger)
    base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    logger.info(f"[TEST] Start test")
    base_model.eval()  # set model to eval mode
    num = 0
    batch = 8
    generated_points, masks = base_model.module.sample(batch)
    ### you can visualize or evaluate the sampled point clouds here
    # write_pointcloud(...)
    pass
