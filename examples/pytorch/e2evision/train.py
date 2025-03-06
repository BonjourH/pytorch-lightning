import os
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.loggers import WandbLogger
import argparse
import warnings
import sys
import time

from base import SourceCameraId
from models.module import E2EPerceptionModule
from e2e_dataset.datamodule import E2EPerceptionDataModule
from utils.visualization import Visualizer
from configs.config import get_config

# Import configuration file
# Configuration file parameters
# Important control parameters
# General way to override any configuration item in the configuration file
# Set random seed
# Get checkpoint path
# Save all checkpoints
# Save the last checkpoint as last.ckpt

def parse_args():
    parser = argparse.ArgumentParser(description='Train E2E perception model')
    
    # Configuration file parameters
    parser.add_argument('--config_file', type=str, default=None, help='config file path, could be a python file, yaml file or json file')
    
    # Important control parameters
    parser.add_argument('--experiment-name', type=str, help='Name of the experiment')
    parser.add_argument('--save_dir', type=str, default='logs', help='Directory to save outputs')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint-path', type=str, help='Specific checkpoint path to resume from')

    # General way to override any configuration item in the configuration file
    parser.add_argument('--config-override', nargs='+', action='append', 
                        help='Override config values. Format: section.key=value')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    config = get_config(args)
    # Set random seed
    L.seed_everything(config.training.seed, workers=True)
    
    # Create data module
    datamodule = E2EPerceptionDataModule(
        train_list=config.training.train_list,
        val_list=config.training.val_list,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        camera_ids=config.data.camera_ids
    )
    
    # Create model
    model = E2EPerceptionModule(
        camera_ids=config.data.camera_ids,
        feature_dim=config.model.feature_dim,
        num_queries=config.model.num_queries,
        num_decoder_layers=config.model.num_decoder_layers,
        backbone=config.model.backbone,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        use_pretrained=config.training.pretrained_weights if not hasattr(args, 'pretrained_weights') or args.pretrained_weights is None else args.pretrained_weights
    )
    
    # Create loggers
    experiment_name = args.experiment_name if args.experiment_name else f"e2e_perception_{time.strftime('%Y%m%d')}"
    save_dir = args.save_dir if args.save_dir else config.logging.log_dir
    
    tb_logger = TensorBoardLogger(
        save_dir=save_dir,
        name=experiment_name,
        default_hp_metric=False
    )
    csv_logger = CSVLogger(
        save_dir=save_dir,
        name=experiment_name
    )
    wandb_logger = WandbLogger(
        project='e2e_perception',
        name=experiment_name,
        save_dir=save_dir
    )
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.logging.checkpoint_dir,
        filename='epoch{epoch:02d}',
        save_top_k=-1,  # Save all checkpoints
        save_last=True  # Save the last checkpoint as last.ckpt
    )
    
    # Get checkpoint path
    checkpoint_path = get_checkpoint_path(config, args)
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        precision=config.training.precision,
        logger=[tb_logger, csv_logger, wandb_logger],
        callbacks=[checkpoint_callback],
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        deterministic=True,
    )
    
    # Train model
    trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)

def get_checkpoint_path(config, args):
    """Get checkpoint path"""
    if args.checkpoint_path:
        return args.checkpoint_path
    
    checkpoint_path = os.path.join(config.logging.checkpoint_dir, config.logging.checkpoint_file)
    if args.resume and os.path.exists(checkpoint_path):
        return checkpoint_path
    
    return None

if __name__ == '__main__':
    main()
    