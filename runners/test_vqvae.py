import torch
import torch.nn as nn
from utils import builder
from utils import misc
import time
from utils.AverageMeter import AverageMeter
from pytorch3d.loss import chamfer_distance
from utils.pointcloud import write_pointcloud
import numpy as np
from einops import rearrange
from torchvision import transforms

def test_vqvae(args, config, logger):
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

    losses = AverageMeter(['loss','rec'])
    with torch.no_grad():
        for idx, (index,point,centers,masks) in enumerate(test_dataloader):
            point = point.cuda().float()
            centers = centers.cuda().float()
            masks = masks.cuda()
            outputs, l, predicted_masks = base_model(point)
            rec_loss = torch.stack([chamfer_distance(point[:,i],outputs[:,i],point_reduction='sum',batch_reduction=None)[0] for i in range(32)],dim=1)
            rec_loss = (rec_loss * masks).mean()
            loss = rec_loss
            losses.update([loss.item(),rec_loss.item()])
            outputs = outputs * masks.unsqueeze(2).unsqueeze(3)
            outputs = rearrange(outputs,'b n p c -> b (n p) c')
            point = rearrange(point,'b n p c -> b (n p) c')

            # for visualization
            # for i in range(point.shape[0]):
            #     write_pointcloud(point[i].cpu().numpy(),f'')
            #     write_pointcloud(outputs[i].cpu().numpy(),f'')

        logger.info('[TEST] Loss rec_loss kl_loss = %s' % (['%.4f' % l for l in losses.avg()]))

