import os
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy

import argparse
import warnings
import sys
import time
import shutil
import wandb  # 添加 wandb 导入

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

# 自定义回调，在每个 epoch 结束后复制 last.ckpt
class CheckpointCopyCallback(Callback):
    """在每个 epoch 结束后复制 last.ckpt 到指定目录"""
    
    def __init__(self, src_dir, dst_dir, max_wait_seconds=30, check_interval=2):
        super().__init__()
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.max_wait_seconds = max_wait_seconds  # 最长等待时间（秒）
        self.check_interval = check_interval  # 检查间隔（秒）
        
    def on_train_epoch_end(self, trainer, pl_module):
        """在每个训练 epoch 结束后执行"""
        if not os.path.exists(self.dst_dir):
            os.makedirs(self.dst_dir, exist_ok=True)
            
        # 检查是否有 last.ckpt
        last_ckpt_src = os.path.join(self.src_dir, 'last.ckpt')
        last_ckpt_dst = os.path.join(self.dst_dir, 'last.ckpt')
        
        # 等待文件写入完成
        print(f"Epoch {trainer.current_epoch}: Waiting for checkpoint file to be ready...")
        
        # 等待文件出现并且大小稳定（表示写入完成）
        wait_start_time = time.time()
        previous_size = -1
        stable_count = 0
        
        while time.time() - wait_start_time < self.max_wait_seconds:
            if not os.path.exists(last_ckpt_src):
                print(f"Waiting for checkpoint file to appear...")
                time.sleep(self.check_interval)
                continue
                
            current_size = os.path.getsize(last_ckpt_src)
            
            # 文件大小稳定检查
            if current_size == previous_size:
                stable_count += 1
                if stable_count >= 2:  # 文件大小连续两次检查都稳定，认为写入完成
                    break
            else:
                stable_count = 0
                previous_size = current_size
                
            print(f"Checkpoint file size: {current_size / (1024*1024):.2f} MB, waiting for size to stabilize...")
            time.sleep(self.check_interval)
            
        # 检查是否超时
        if time.time() - wait_start_time >= self.max_wait_seconds:
            print(f"Warning: Reached maximum wait time of {self.max_wait_seconds} seconds. Proceeding with copy.")
        
        # 最终检查和复制
        if os.path.exists(last_ckpt_src):
            print(f"Epoch {trainer.current_epoch}: Copying last checkpoint to {last_ckpt_dst}")
            shutil.copy2(last_ckpt_src, last_ckpt_dst)
            # os.remove(last_ckpt_src)
            print(f"Checkpoint copied successfully")
        else:
            print(f"Warning: Last checkpoint not found at {last_ckpt_src} after waiting")

def clean_wandb_history(project_name):
    """清理 W&B 上该项目的所有历史运行数据"""
    try:
        api = wandb.Api()
        runs = api.runs(project_name)
        
        print(f"正在清理 {project_name} 项目的历史数据...")
        for run in runs:
            print(f"删除运行: {run.name} (ID: {run.id})")
            run.delete()
        
        print(f"成功清理了 {len(runs)} 个历史运行")
    except Exception as e:
        print(f"清理 W&B 历史数据时出错: {e}")
        print("继续训练，但不清理历史数据")

def parse_args():
    parser = argparse.ArgumentParser(description='Train E2E perception model')
    
    # Configuration file parameters
    parser.add_argument('--config_file', type=str, default=None, help='config file path, could be a python file, yaml file or json file')
    
    # Important control parameters
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment')
    parser.add_argument('--validate_only', action='store_true', help='Only run validation, no training')
    parser.add_argument('--resume', type=bool, default=False, help='Resume training from the last checkpoint')
    
    # General way to override any configuration item in the configuration file
    parser.add_argument('--config-override', nargs='+', action='append', 
                        help='Override config values. Format: section.key=value')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    config = get_config(args)
    
    # 如果配置为清理 W&B 历史数据且启用了 W&B
    if config.logging.use_wandb and config.logging.clean_wandb_history:
        clean_wandb_history(config.logging.wandb_project)
    
    # Set random seed
    L.seed_everything(config.training.seed, workers=True)
    
    # Create data module
    datamodule = E2EPerceptionDataModule(
        train_list=config.training.train_list,
        val_list=config.training.val_list,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        data_config=config.data
    )
    
    # Create model
    model = E2EPerceptionModule(config=config)
    
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

    # Add custom last checkpoint path if specified
    if os.path.exists(config.logging.last_checkpoint_dir):
        checkpoint_callback.last_checkpoint_path = os.path.join(config.logging.last_checkpoint_dir, 'last.ckpt')
    
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="step"),
        FilteredProgressBar(
            refresh_rate=1,
            metrics_to_display=config.logging.progress_bar_metrics
        )
    ]
    
    # 添加自定义 checkpoint 复制回调
    if hasattr(config.logging, 'last_checkpoint_dir') and config.logging.last_checkpoint_dir:
        copy_callback = CheckpointCopyCallback(
            src_dir=config.logging.checkpoint_dir,
            dst_dir=config.logging.last_checkpoint_dir
        )
        callbacks.append(copy_callback)
        print(f"Added checkpoint copy callback to copy last.ckpt to {config.logging.last_checkpoint_dir} after each epoch")
    
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
        enable_progress_bar=True,
        limit_val_batches=config.training.limit_val_batches,
        log_every_n_steps=config.training.log_every_n_steps
    )
    
    # Get checkpoint path
    checkpoint_path = os.path.join(config.logging.checkpoint_dir, 'last.ckpt')
    
    if os.path.exists(checkpoint_path) and args.resume:
        if args.validate_only:
            print("Running validation only...")
            trainer.validate(model, datamodule=datamodule, ckpt_path=checkpoint_path)
        else:
            print("Starting training...")
            trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    main()
    