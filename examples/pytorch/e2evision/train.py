import os
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger, WandbLogger
import argparse
import warnings
import sys
import time

from base import SourceCameraId
from models.module import E2EPerceptionModule
from e2e_dataset.datamodule import E2EPerceptionDataModule
from utils.visualization import Visualizer
from utils.metrics import E2EPerceptionWandbLogger, FilteredProgressBar
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
    # General way to override any configuration item in the configuration file
    parser.add_argument('--experiment_name', type=str, default=None, help='Name of the experiment')
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
        use_pretrained=config.training.pretrained_weights
    )
    
    # Create loggers
    experiment_name = args.experiment_name if args.experiment_name else f"e2e_perception_{time.strftime('%Y%m%d')}"

    log_dir = config.logging.log_dir
    loggers = []
    if config.logging.use_tensorboard:
        loggers.append(TensorBoardLogger(save_dir=log_dir, name=experiment_name, default_hp_metric=False))
    if config.logging.use_csv:
        loggers.append(CSVLogger(save_dir=log_dir, name=experiment_name))
    if config.logging.use_wandb:
        loggers.append(E2EPerceptionWandbLogger(
            project=config.logging.wandb_project,
            name=experiment_name,
            save_dir=log_dir,
            keys_to_log=config.logging.wandb_log_metrics,
            use_optional_metrics=config.logging.use_optional_metrics
        ))
        
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.logging.checkpoint_dir,
        filename='epoch{epoch:02d}',
        save_top_k=config.logging.save_top_k,
        save_last=True,
        monitor='train/loss_epoch'
    )
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="step"),
        FilteredProgressBar(
            refresh_rate=1,
            metrics_to_display=config.logging.progress_bar_metrics
        )
    ]
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        precision=config.training.precision,
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        deterministic=True,
    )
    
    # Train model
    checkpoint_path = os.path.join(config.logging.checkpoint_dir, 'last.ckpt')
    trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)

if __name__ == '__main__':
    main()
    