import os, sys
import torch
import torch.optim as optim
from utils import registry
from utils.misc import *
from timm.scheduler import CosineLRScheduler
MODELS = registry.Registry('models')
DATASETS = registry.Registry('dataset')


def build_dataset_from_cfg(cfg, default_args = None):
    return DATASETS.build(cfg, default_args = default_args)


def build_model_from_cfg(cfg, **kwargs):
    return MODELS.build(cfg, **kwargs)

def dataset_builder(args, config):
    dataset = build_dataset_from_cfg(config)
    shuffle = config.train
    sampler = None
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.bs,
                                            shuffle = shuffle, 
                                            drop_last = config.train,
                                            num_workers = int(args.num_workers),
                                            worker_init_fn=worker_init_fn)
    return sampler, dataloader

def model_builder(config):
    model = build_model_from_cfg(config)
    return model

def build_opti_sche(base_model, config):
    opti_config = config.optimizer
    if opti_config.type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    # print(name)
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]
        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(base_model.parameters(), **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(base_model.parameters(), nesterov=True, **opti_config.kwargs)
    else:
        raise NotImplementedError()

    sche_config = config.scheduler
    if sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                t_initial=sche_config.kwargs.epochs,
                lr_min=1e-6,
                warmup_lr_init=1e-6,
                warmup_t=sche_config.kwargs.initial_epochs,
                cycle_limit=1,
                t_in_epochs=True)
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_config.kwargs)
    elif sche_config.type == 'function':
        scheduler = None
    else:
        raise NotImplementedError()
    
    return optimizer, scheduler

def save_checkpoint(base_model, optimizer, epoch, best_metrics, prefix, args, logger = None):
    torch.save({
                'base_model' : base_model.module.state_dict() ,
                'optimizer' : optimizer.state_dict(),
                'epoch' : epoch,
                'best_metrics' : best_metrics,
                }, os.path.join(args.experiment_path, prefix + '.pth'))
    logger.info(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}")

def load_model_from_ckpt(model,ckpt_path):
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict['base_model'],strict=True)