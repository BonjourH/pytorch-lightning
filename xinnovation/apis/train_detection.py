import os
import torch
import argparse
from pathlib import Path
from xinnovation.src.core.config import Config
from xinnovation.src.core.registry import (
    BACKBONES, NECKS, HEADS, LOSSES,
    DATASETS, TRANSFORMS, SAMPLERS, LOADERS,
    OPTIMIZERS, SCHEDULERS, CALLBACKS, LOGGERS,
    STRATEGIES
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a detection model')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--gpu-ids', type=int, nargs='+', help='ids of gpus to use')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    return parser.parse_args()

def build_model(cfg):
    """Build model from config."""
    # Build backbone
    backbone = BACKBONES.build(cfg.model.backbone)
    
    # Build neck
    neck = NECKS.build(cfg.model.neck)
    
    # Build head
    head = HEADS.build(cfg.model.head)
    
    # Build loss
    loss = LOSSES.build(cfg.model.loss)
    
    # Combine into model
    model = DetectionModel(
        backbone=backbone,
        neck=neck,
        head=head,
        loss=loss
    )
    
    return model

def build_dataset(cfg, default_args=None):
    """Build dataset from config."""
    if default_args is None:
        default_args = {}
        
    # Build transforms
    transforms = []
    for t in cfg.transforms:
        transforms.append(TRANSFORMS.build(t))
        
    # Build dataset
    dataset = DATASETS.build(
        cfg,
        default_args=dict(
            transforms=transforms,
            **default_args
        )
    )
    
    return dataset

def build_dataloader(dataset, cfg, default_args=None):
    """Build dataloader from config."""
    if default_args is None:
        default_args = {}
        
    # Build sampler
    sampler = SAMPLERS.build(
        cfg.sampler,
        default_args=dict(dataset=dataset, **default_args)
    )
    
    # Build dataloader
    dataloader = LOADERS.build(
        cfg,
        default_args=dict(
            dataset=dataset,
            sampler=sampler,
            **default_args
        )
    )
    
    return dataloader

def build_optimizer(cfg, model):
    """Build optimizer from config."""
    optimizer = OPTIMIZERS.build(
        cfg,
        default_args=dict(params=model.parameters())
    )
    return optimizer

def build_scheduler(cfg, optimizer):
    """Build scheduler from config."""
    scheduler = SCHEDULERS.build(
        cfg,
        default_args=dict(optimizer=optimizer)
    )
    return scheduler

def build_callbacks(cfg):
    """Build callbacks from config."""
    callbacks = []
    for c in cfg:
        callbacks.append(CALLBACKS.build(c))
    return callbacks

def build_loggers(cfg):
    """Build loggers from config."""
    loggers = []
    for l in cfg:
        loggers.append(LOGGERS.build(l))
    return loggers

def build_strategy(cfg):
    """Build training strategy from config."""
    strategy = STRATEGIES.build(cfg)
    return strategy

def main():
    """Main training function."""
    args = parse_args()
    
    # Load config
    cfg = Config(args.config)
    
    # Set work directory
    if args.work_dir is not None:
        cfg.train.work_dir = args.work_dir
    work_dir = Path(cfg.train.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Set GPU
    if args.gpu_ids is not None:
        cfg.train.gpu_ids = args.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build model
    model = build_model(cfg)
    model = model.to(device)
    
    # Build datasets
    train_dataset = build_dataset(cfg.data.train)
    val_dataset = build_dataset(cfg.data.val)
    
    # Build dataloaders
    train_loader = build_dataloader(
        train_dataset,
        cfg.train,
        default_args=dict(num_workers=cfg.train.num_workers)
    )
    val_loader = build_dataloader(
        val_dataset,
        cfg.train,
        default_args=dict(num_workers=cfg.train.num_workers)
    )
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(cfg.train.optimizer, model)
    scheduler = build_scheduler(cfg.train.scheduler, optimizer)
    
    # Build callbacks and loggers
    callbacks = build_callbacks(cfg.train.callbacks)
    loggers = build_loggers(cfg.train.loggers)
    
    # Build training strategy
    strategy = build_strategy(cfg.train.strategy)
    
    # Create trainer
    trainer = DetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=callbacks,
        loggers=loggers,
        strategy=strategy,
        work_dir=work_dir,
        device=device,
        **cfg.train
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        trainer.resume_from_checkpoint(args.resume_from)
    
    # Start training
    trainer.fit()

if __name__ == '__main__':
    main() 