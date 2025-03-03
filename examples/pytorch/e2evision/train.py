import os
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from typing import Dict, List, Any
import warnings

from base import SourceCameraId
from data import MultiFrameDataset, custom_collate_fn
from model import E2EPerceptionNet
from loss import TrajectoryLoss

class E2EPerceptionModule(L.LightningModule):
    """Lightning module for end-to-end perception."""
    def __init__(self, 
                 camera_ids: List[SourceCameraId],
                 feature_dim: int = 256,
                 num_queries: int = 100,
                 num_decoder_layers: int = 6,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 sequence_length: int = 10,
                 batch_size: int = 2,
                 num_workers: int = 4,
                 train_list: str = None,
                 val_list: str = None):
        super().__init__()
        self.save_hyperparameters(ignore=['camera_ids'])
        self.camera_ids = camera_ids
        
        # Create model
        self.model = E2EPerceptionNet(
            camera_ids=camera_ids,
            feature_dim=feature_dim,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers
        )
        
        # Create loss function
        self.criterion = TrajectoryLoss()
        
    def print_memory_usage(self, step: str):
        """Helper to print memory usage."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            self.print(f"{step}:")
            self.print(f"  Allocated memory: {memory_allocated:.2f} MB")
            self.print(f"  Reserved memory: {memory_reserved:.2f} MB")
    
    def on_fit_start(self):
        """Called when fit begins."""
        self.print_memory_usage("Initial model state")
        
    def forward(self, batch: Dict) -> List[Dict]:
        return self.model(batch)
    
    def training_step(self, batch: Dict, batch_idx: int) -> Dict[str, torch.Tensor]:
        # Forward pass
        outputs = self(batch)
        
        # Calculate loss
        loss_dict = self.criterion(outputs, batch)
        loss = loss_dict['loss']
        
        # Log losses
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            if k != 'loss':
                self.log(f'train_{k}', v, on_step=True, on_epoch=True)
        
        # Print memory usage occasionally
        if self.global_step > 0 and self.global_step % 100 == 0:
            self.print_memory_usage(f"Training step {self.global_step}")
            
        return loss
    
    def validation_step(self, batch: Dict, batch_idx: int) -> Dict[str, torch.Tensor]:
        # Forward pass
        outputs = self(batch)
        
        # Calculate loss
        loss_dict = self.criterion(outputs, batch)
        loss = loss_dict['loss']
        
        # Log losses
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            if k != 'loss':
                self.log(f'val_{k}', v, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        # Create optimizer
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Create learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.learning_rate * 0.01
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def setup(self, stage: str):
        """Setup datasets and dataloaders."""
        if stage == "fit":
            # Read train and validation clip lists
            train_clips = read_clip_list(self.hparams.train_list)
            val_clips = read_clip_list(self.hparams.val_list)
            
            self.print(f"Found {len(train_clips)} training clips and {len(val_clips)} validation clips")
            
            # Create datasets
            self.train_dataset = MultiFrameDataset(
                clip_dirs=train_clips,
                camera_ids=self.camera_ids,
                sequence_length=self.hparams.sequence_length
            )
            
            self.val_dataset = MultiFrameDataset(
                clip_dirs=val_clips,
                camera_ids=self.camera_ids,
                sequence_length=self.hparams.sequence_length
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True
        )

def parse_args():
    parser = argparse.ArgumentParser(description='Train E2E perception model')
    
    # Data parameters
    parser.add_argument('--train-list', type=str, required=True, help='Path to txt file containing training clip paths')
    parser.add_argument('--val-list', type=str, required=True, help='Path to txt file containing validation clip paths')
    parser.add_argument('--sequence-length', type=int, default=10, help='Number of frames in each sequence')
    
    # Model parameters
    parser.add_argument('--feature-dim', type=int, default=256, help='Feature dimension')
    parser.add_argument('--num-queries', type=int, default=100, help='Number of object queries')
    parser.add_argument('--num-decoder-layers', type=int, default=6, help='Number of decoder layers')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    
    # Trainer parameters
    parser.add_argument('--accelerator', type=str, default='auto', help='Accelerator to use (auto, gpu, cpu)')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices to use')
    parser.add_argument('--precision', type=str, default='32-true', help='Precision for training')
    parser.add_argument('--strategy', type=str, default='auto', help='Strategy to use for distributed training')
    
    # Logging and saving
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    
    return parser.parse_args()

def read_clip_list(list_file: str) -> List[str]:
    """Read clip paths from a txt file."""
    if not os.path.exists(list_file):
        raise FileNotFoundError(f"Clip list file not found: {list_file}")
        
    with open(list_file, 'r') as f:
        clips = [line.strip() for line in f.readlines() if line.strip()]
        
    # Verify all paths exist
    for clip_path in clips:
        if not os.path.exists(clip_path):
            raise FileNotFoundError(f"Clip directory not found: {clip_path}")
            
    return clips

def main():
    args = parse_args()
    
    # Define camera IDs
    camera_ids = [
        SourceCameraId.FRONT_CENTER_CAMERA,
        SourceCameraId.FRONT_LEFT_CAMERA,
        SourceCameraId.FRONT_RIGHT_CAMERA,
        SourceCameraId.SIDE_LEFT_CAMERA,
        SourceCameraId.SIDE_RIGHT_CAMERA,
        SourceCameraId.REAR_LEFT_CAMERA,
        SourceCameraId.REAR_RIGHT_CAMERA
    ]
    
    # Create model
    model = E2EPerceptionModule(
        camera_ids=camera_ids,
        feature_dim=args.feature_dim,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_list=args.train_list,
        val_list=args.val_list
    )
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename='e2e_perception-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_last=True,
        save_top_k=3
    )
    
    # Create logger
    try:
        logger = TensorBoardLogger(
            save_dir=os.path.dirname(args.save_dir),
            name=os.path.basename(args.save_dir)
        )
    except ModuleNotFoundError:
        warnings.warn(
            "TensorBoard not available. Using CSVLogger instead. "
            "To use TensorBoard, run: pip install tensorboard"
        )
        logger = CSVLogger(
            save_dir=args.save_dir,
            name='logs'
        )
    
    # Create trainer
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        strategy=args.strategy,
        max_epochs=args.num_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        gradient_clip_val=0.1,
        log_every_n_steps=10,
        deterministic=True
    )
    
    # Train model
    trainer.fit(
        model,
        ckpt_path=args.resume if args.resume else None
    )

if __name__ == '__main__':
    main() 