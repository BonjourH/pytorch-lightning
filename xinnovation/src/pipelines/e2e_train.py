"""
End-to-end training script for perception tasks.
"""
import os
import argparse
from typing import Dict, Any
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from lightning_modules import BaseModule
from datasets import CustomDataset
from transforms import Compose, Resize, RandomFlip, Normalize, ToTensor
from callbacks import DetectionVisualization
from registry import MODELS, DATASETS, TRANSFORMS, CALLBACKS, LOSSES


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train a perception model')
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('--work-dir', help='Work directory', default='work_dirs')
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--gpus', type=int, help='Number of GPUs to use', default=1)
    parser.add_argument('--max-epochs', type=int, help='Maximum epochs', default=100)
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_transforms(transform_cfg: Dict[str, Any]) -> Compose:
    """Build data transformation pipeline."""
    transforms = []
    for t in transform_cfg:
        transform_type = t.pop('type')
        transform = TRANSFORMS.get(transform_type)(**t)
        transforms.append(transform)
    return Compose(transforms)


def build_dataloader(data_cfg: Dict[str, Any], transform: Compose) -> Dict[str, Any]:
    """Build data loaders for training, validation, and testing."""
    data_loaders = {}
    
    for split in ['train', 'val', 'test']:
        if f'{split}_data' in data_cfg:
            dataset_cfg = data_cfg[f'{split}_data']
            dataset_cfg['pipeline'] = transform
            
            # Create dataset
            dataset = DATASETS.get(dataset_cfg.pop('type'))(**dataset_cfg)
            
            # Create dataloader
            loader_cfg = data_cfg[f'{split}_dataloader']
            data_loaders[split] = torch.utils.data.DataLoader(
                dataset,
                batch_size=loader_cfg.get('batch_size', 32),
                shuffle=(split == 'train'),
                num_workers=loader_cfg.get('num_workers', 4),
                pin_memory=True
            )
    
    return data_loaders


def build_callbacks(callback_cfg: Dict[str, Any], work_dir: str) -> List[pl.Callback]:
    """Build training callbacks."""
    callbacks = []
    
    # Add ModelCheckpoint callback
    ckpt_callback = ModelCheckpoint(
        dirpath=os.path.join(work_dir, 'checkpoints'),
        filename='epoch_{epoch:03d}-val_loss_{val/loss:.4f}',
        monitor='val/loss',
        save_top_k=3,
        mode='min',
        save_last=True
    )
    callbacks.append(ckpt_callback)
    
    # Add LearningRateMonitor
    lr_callback = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_callback)
    
    # Add custom callbacks from config
    for cb_name, cb_cfg in callback_cfg.items():
        callback_type = cb_cfg.pop('type')
        callback = CALLBACKS.get(callback_type)(**cb_cfg)
        callbacks.append(callback)
    
    return callbacks


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Create work directory
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Build transforms
    transform = build_transforms(config['transform'])
    
    # Build dataloaders
    data_loaders = build_dataloader(config['data'], transform)
    
    # Build model
    model = BaseModule(
        backbone_cfg=config['model']['backbone'],
        neck_cfg=config.get('model', {}).get('neck'),
        head_cfg=config['model']['head'],
        loss_cfg=config['model']['loss'],
        optimizer_cfg=config['optimizer'],
        lr_scheduler_cfg=config.get('lr_scheduler'),
        data_cfg=config['data'],
        model_cfg=config.get('model', {}).get('extra_config')
    )
    
    # Build callbacks
    callbacks = build_callbacks(config.get('callbacks', {}), args.work_dir)
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=args.work_dir,
        name='logs',
        default_hp_metric=False
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        accelerator='gpu' if args.gpus > 0 else None,
        strategy='ddp' if args.gpus > 1 else None,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=1.0,
        num_sanity_val_steps=2,
    )
    
    # Train model
    trainer.fit(
        model,
        train_dataloaders=data_loaders['train'],
        val_dataloaders=data_loaders.get('val')
    )
    
    # Test model if test dataloader exists
    if 'test' in data_loaders:
        trainer.test(model, dataloaders=data_loaders['test'])


if __name__ == '__main__':
    main() 