import torch
import torch.nn as nn
from utils import builder
import time
from utils.AverageMeter import AverageMeter
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import rotation_6d_to_matrix
from einops import rearrange

def pairwise_min_distances_adjacent_vectorized(P_batch, masks, eps=1e-6):
    B, N, P, _ = P_batch.shape
    def get_valid_adjacent_pairs(masks):
        pairs = []
        for b in range(B):
            valid = torch.nonzero(masks[b], as_tuple=False).squeeze(-1)
            valid = valid[~((valid == 15) | (valid == 31))]  
            if valid.numel() < 2:
                continue
            i = valid[:-1]
            j = valid[1:]
            b_idx = torch.full_like(i, b)
            pairs.append(torch.stack([b_idx, i, j], dim=1))
        if not pairs:
            return None, None, None
        all_pairs = torch.cat(pairs, dim=0)
        return all_pairs[:, 0], all_pairs[:, 1], all_pairs[:, 2]  

    b_idx, i_idx, j_idx = get_valid_adjacent_pairs(masks)

    if b_idx is None:
        return torch.tensor(0.0, device=P_batch.device)
    
    pc1 = P_batch[b_idx, i_idx]  
    pc2 = P_batch[b_idx, j_idx]  
    diff = pc1[:, :, None, :] - pc2[:, None, :, :] 
    dists = torch.norm(diff, dim=-1)               
    min_dists = torch.amin(dists, dim=(1, 2)) + eps  
    c1 = torch.mean(pc1, dim=1)
    c2 = torch.mean(pc2, dim=1)
    center_dists = torch.norm(c1 - c2, dim=-1)
    loss = (min_dists / (min_dists + center_dists)) ** 2

    return loss.sum()


def train_transformer(args, config, train_writer, val_writer, logger):
    # build dataset
    config.model.args = args
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    
    # parameter setting
    start_epoch = 0
    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
    else:
        if args.ckpts != '':
            # base_model.load_model_from_ckpt(args.ckpts)
            checkpoints = torch.load(args.ckpts)
            base_model.load_state_dict(checkpoints['base_model'])
        else:
            logger.info('Training from scratch')
    best_metrics = 99999
 
    logger.info('Using Data parallel ...' , logger = logger)
    device = torch.device('cuda') 
    base_model = nn.DataParallel(base_model).to(device)
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    for epoch in range(start_epoch, config.max_epoch + 1):
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss','rec_loss','col_loss'])
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        for idx, (index,before_points,before_centroid,after_points,after_centroid,masks) in enumerate(train_dataloader):      #TODO
            optimizer.zero_grad()
            data_time.update(time.time() - batch_start_time)

            #####TODO
            before_points = before_points.cuda().float()
            before_centroid = before_centroid.cuda().float()
            after_points = after_points.cuda().float()
            after_centroid = after_centroid.cuda().float()
            masks = masks.cuda()
            batch = masks.shape[0]
            predicted_param = base_model(before_points, before_centroid, after_points, masks)

            rotation_6d = predicted_param[:,:,:6]
            transition =  predicted_param[:,:,6:]
            rotation_6d = rearrange(rotation_6d,'b n c -> (b n) c')
            rot_matrix = rotation_6d_to_matrix(rotation_6d)
            after_points = rearrange(after_points,'b n p c -> (b n) p c')
            after_centroid = rearrange(after_centroid,'b n c -> (b n) c')
            transition = rearrange(transition,'b n c -> (b n) c')
            generated_points = torch.bmm(after_points - after_centroid.unsqueeze(1),rot_matrix) + (transition+after_centroid).unsqueeze(1)
            generated_points = rearrange(generated_points,'(b n) p c -> b n p c',n=32)
            col_loss = pairwise_min_distances_adjacent_vectorized(generated_points, masks)

            # diffusion loss
            criterion = nn.MSELoss(reduction='none')
            rec_loss = (criterion(generated_points,before_points)*masks[:,:,None,None]).sum()
            loss = rec_loss + col_loss
            losses.update([loss.item(),rec_loss.item(),col_loss.item()])

            # collision loss

            #######

            loss.backward()
            optimizer.step()
            losses.update([loss.item()])

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            
            if idx % 5 == 0:
                logger.info('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss rec_loss= %s lr = %.6f' %
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

        # print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
        #     (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],optimizer.param_groups[0]['lr']), logger = logger)
        logger.info('[Training] EPOCH: %d EpochTime = %.3f (s) Loss diff_loss rec_loss = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],optimizer.param_groups[0]['lr']))

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)

            # better = metrics.better_than(best_metrics)
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
    losses = AverageMeter(['loss','rec_loss'])
    with torch.no_grad():
        for idx, (index,before_points,before_centroid,after_points,after_centroid,masks) in enumerate(test_dataloader):
            before_points = before_points.cuda().float()
            before_centroid = before_centroid.cuda().float()
            after_points = after_points.cuda().float()
            after_centroid = after_centroid.cuda().float()
            masks = masks.cuda()
            batch = masks.shape[0]
            # predicted_param = base_model(after_points, after_centroid,before_points, masks)
            predicted_param = base_model(before_points, before_centroid, after_points, masks)

            # transform
            rotation_6d = predicted_param[:,:,:6]
            transition =  predicted_param[:,:,6:]
            rotation_6d = rearrange(rotation_6d,'b n c -> (b n) c')
            rot_matrix = rotation_6d_to_matrix(rotation_6d)
            after_points = rearrange(after_points,'b n p c -> (b n) p c')
            after_centroid = rearrange(after_centroid,'b n c -> (b n) c')
            transition = rearrange(transition,'b n c -> (b n) c')
            generated_points = torch.bmm(after_points - after_centroid.unsqueeze(1),rot_matrix) + (transition+after_centroid).unsqueeze(1)
            generated_points = rearrange(generated_points,'(b n) p c -> b n p c',n=32)

            # diffusion loss
            criterion = nn.MSELoss(reduction='none')
            rec_loss = (10*criterion(generated_points,before_points)*masks[:,:,None,None]).sum()/batch
            loss = rec_loss
            losses.update([loss.item(),rec_loss.item()])

        logger.info('[Validation] EPOCH: %d  Loss rec_loss = %s' % (epoch,['%.4f' % l for l in losses.avg()]))


    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/loss', losses.avg(0), epoch)

    return losses.avg()[0]
