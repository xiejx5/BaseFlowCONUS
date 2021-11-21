import hydra
import torch
import logging
import torch.distributed as dist
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from source.trainer import Trainer
from hydra.utils import instantiate
# from source.utils.util import instantiate

import os
from hydra.experimental import compose, initialize_config_dir
initialize_config_dir(config_dir=os.path.join(
    os.getcwd(), "configs"), job_name="test_app")
cfg = compose(config_name="config", overrides=["work_dir=."])


def train_worker(cfg):
    logger = logging.getLogger('trainer')
    # setup data_loader instances
    data_loader, valid_loader = instantiate(cfg.dataloader)
    train_loader, valid_data_loader = instantiate(cfg.dataloader)

    # build model. print it's structure and # trainable params.
    model = instantiate(cfg.arch)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    logger.info(model)
    logger.info(
        f'Trainable parameters: {sum([p.numel() for p in trainable_params])}')

    # get function handles of loss and metrics
    criterion = instantiate(cfg.loss, is_func=True)
    metrics = [instantiate(met, is_func=True) for met in cfg['metrics']]

    # build optimizer, learning rate scheduler.
    optimizer = instantiate(cfg.optimizer, model.parameters())
    lr_scheduler = instantiate(cfg.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=cfg,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()


def init_worker(rank, ngpus, working_dir, config):
    # initialize training config
    config = OmegaConf.create(config)
    config.local_rank = rank
    config.cwd = working_dir
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:34567',
        world_size=ngpus,
        rank=rank)
    torch.cuda.set_device(rank)

    # start training processes
    train_worker(config)


def train(cfg):
    n_gpu = torch.cuda.device_count()
    assert n_gpu, 'Can\'t find any GPU device on this machine.'

    working_dir = str(Path.cwd().relative_to(hydra.utils.get_original_cwd()))

    if cfg.resume is not None:
        cfg.resume = hydra.utils.to_absolute_path(cfg.resume)
    cfg = OmegaConf.to_yaml(cfg, resolve=True)
    torch.multiprocessing.spawn(
        init_worker, nprocs=n_gpu, args=(n_gpu, working_dir, cfg))
