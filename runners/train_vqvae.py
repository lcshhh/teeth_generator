import torch
import torch.nn as nn
from utils import builder
import time
from utils.AverageMeter import AverageMeter
from pytorch3d.loss import chamfer_distance
from einops import rearrange
from torch import autograd

import numpy as np
from torchvision import transforms

def train_vqvae(args, config, train_writer, val_writer, logger):
    # build dataset
    config.model.args = args
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    
    # parameter setting
    start_epoch = 0
    best_metrics = 99999
    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
    else:
        if args.ckpts != '':
            checkpoints = torch.load(args.ckpts)
            base_model.load_state_dict(checkpoints['base_model'])
        else:
            logger.info('Training from scratch')
 
    logger.info('Using Data parallel ...' , logger = logger)
    device = torch.device('cuda') 
    base_model = nn.DataParallel(base_model).to(device)
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # training
    for epoch in range(start_epoch, config.max_epoch + 1):
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        losses = AverageMeter(['loss','rec_loss','ent_loss','mask_loss'])
        for idx, (index,point,centers,masks) in enumerate(train_dataloader): 
            optimizer.zero_grad()
            data_time.update(time.time() - batch_start_time)

            #####
            point = point.cuda().float()
            centers = centers.cuda().float()
            masks = masks.cuda()
            outputs,entropy_loss,predicted_masks = base_model(point)
            entropy_loss = entropy_loss.mean()
            rec_loss = torch.stack([chamfer_distance(point[:,i],outputs[:,i],point_reduction='sum',batch_reduction=None)[0] for i in range(32)],dim=1)
            criterion = nn.MSELoss()
            rec_loss = 10 * (rec_loss * masks).mean()
            mask_loss = criterion(masks.float(),predicted_masks)
            loss = rec_loss + entropy_loss + mask_loss
            #######
            loss.backward()
            optimizer.step()
            losses.update([loss.item(),rec_loss.item(),entropy_loss.item(),mask_loss.item()])

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            
            if idx % 5 == 0:
                logger.info('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss rec_loss ent_loss mask = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']))
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        logger.info('[Training] EPOCH: %d EpochTime = %.3f (s) Loss kl_loss rec_loss ent_loss = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],optimizer.param_groups[0]['lr']))

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
            # Save ckeckpoints
            if metrics < best_metrics:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, best_metrics, 'ckpt-best', args, logger = logger)
                logger.info("--------------------------------------------------------------------------------------------")

        builder.save_checkpoint(base_model, optimizer, epoch, best_metrics, 'ckpt-last', args, logger = logger)      
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger = None):
    logger.info(f"[VALIDATION] Start validating epoch {epoch}")
    base_model.eval()  # set model to eval mode
    losses = AverageMeter(['loss','rec_loss','mask_loss'])
    with torch.no_grad():
        for idx, (index,point,centers,masks) in enumerate(test_dataloader):
            point = point.cuda().float()
            centers = centers.cuda().float()
            masks = masks.cuda()
            outputs,entropy_loss,predicted_masks = base_model(point)
            entropy_loss = entropy_loss.mean()
            rec_loss = 10 * torch.stack([chamfer_distance(point[:,i],outputs[:,i],point_reduction='sum',batch_reduction=None)[0] for i in range(32)],dim=1)
            rec_loss = (rec_loss * masks).mean()
            criterion = nn.MSELoss()
            mask_loss = criterion(masks.float(),predicted_masks)
            loss = rec_loss + mask_loss
            losses.update([loss.item(),rec_loss.item(),mask_loss.item()])

        logger.info('[Validation] EPOCH: %d  Loss rec_loss mask_loss = %s' % (epoch,['%.4f' % l for l in losses.avg()]))


    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/loss', losses.avg(0), epoch)

    return losses.avg()[0]