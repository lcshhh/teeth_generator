import torch
import torch.nn as nn
from utils import builder
from utils import misc
import time
from utils.AverageMeter import AverageMeter
from pytorch3d.loss import chamfer_distance
from utils.pointcloud import write_pointcloud
import numpy as np
import vedo
import trimesh
import os
from einops import rearrange
from pytorch3d.transforms import matrix_to_quaternion, Transform3d,rotation_6d_to_matrix
from torchvision import transforms
import torch.nn.functional as F
def test_transformer(args, config, logger):
    # build dataset
    config.model.args = args
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    

    # resume ckpts
    builder.load_model_from_ckpt(base_model,args.ckpts)
 
    logger.info('Using Data parallel ...' , logger = logger)
    device = torch.device('cuda') 
    base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    logger.info(f"[TEST] Start test")
    base_model.eval()  # set model to eval mode

    losses = AverageMeter(['loss'])
    with torch.no_grad():
        for idx, (index,before_points,before_centroid,after_points,after_centroid,masks) in enumerate(test_dataloader):
            before_points = before_points.cuda().float()
            before_centroid = before_centroid.cuda().float()
            after_points = after_points.cuda().float()
            after_centroid = after_centroid.cuda().float()
            masks = masks.cuda()
            predicted_param = base_model(before_points, before_centroid, after_points, masks)
            rotation_6d = predicted_param[:,:,:6]
            transition =  predicted_param[:,:,6:]
            rotation_6d = rearrange(rotation_6d,'b n c -> (b n) c')
            rot_matrix = rotation_6d_to_matrix(rotation_6d)
            after_points = rearrange(after_points,'b n p c -> (b n) p c')
            after_centroid = rearrange(after_centroid,'b n c -> (b n) c')
            transition = rearrange(transition,'b n c -> (b n) c')
            points = torch.bmm(after_points - after_centroid.unsqueeze(1),rot_matrix) + (transition+after_centroid).unsqueeze(1)
            points = rearrange(points,'(b n) p c -> b (n p) c',n=32)
            after_points = rearrange(after_points,'(b n) p c -> b (n p) c',n=32)
            before_points = rearrange(before_points,'b n p c -> b (n p) c',n=32) 
            
            ### for visualization
            # for i in range(before_points.shape[0]):
            #     write_pointcloud(points[i].cpu().numpy(),f'predict.ply')
            #     write_pointcloud(after_points[i].cpu().numpy(),f'after.ply')
            #     write_pointcloud(before_points[i].cpu().numpy(),f'before.ply')
            #     exit()

def sample_transformer(args, config, logger):
    # build dataset
    config.model.args = args
    (_, test_dataloader)= builder.dataset_builder(args, config.dataset.test)
    # build model
    base_model = builder.model_builder(config.model)
    

    # resume ckpts
    builder.load_model_from_ckpt(base_model,args.ckpts)
 
    logger.info('Using Data parallel ...' , logger = logger)
    device = torch.device('cuda') 
    base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    logger.info(f"[TEST] Start test")
    base_model.eval()  # set model to eval mode
    with torch.no_grad():
        for idx, (index,before_points,before_centroid,after_points,after_centroid,masks) in enumerate(test_dataloader):
            batch = index.shape[0]
            before_points = before_points.cuda().float()
            before_centroid = before_centroid.cuda().float()
            after_points = after_points.cuda().float()
            after_centroid = after_centroid.cuda().float()
            masks = masks.cuda()
            points = []
            predicted_param = base_model(before_points, before_centroid, after_points, masks)
            rotation_6d = predicted_param[:,:,:6]
            transition =  predicted_param[:,:,6:]
            rotation_6d = rearrange(rotation_6d,'b n c -> (b n) c')
            rot_matrix = rotation_6d_to_matrix(rotation_6d)
            after_points = rearrange(after_points,'b n p c -> (b n) p c')
            after_centroid = rearrange(after_centroid,'b n c -> (b n) c')
            transition = rearrange(transition,'b n c -> (b n) c')
            points = torch.bmm(after_points - after_centroid.unsqueeze(1),rot_matrix) + (transition+after_centroid).unsqueeze(1)
            points = rearrange(points,'(b n) p c -> b n p c',n=32)
            after_points = rearrange(after_points,'(b n) p c -> b n p c',n=32)

            ### for output
            for i in range(batch):
                for j in range(32):
                    if masks[i][j] > 0.5:
                        write_pointcloud(points[i][j].cpu().numpy(),f'/path/to/your/output_directory/{index[i]}_{j}.ply')
            
