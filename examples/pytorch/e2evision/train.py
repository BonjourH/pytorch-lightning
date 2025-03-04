import os
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
import argparse
import warnings

from base import SourceCameraId
from models.module import E2EPerceptionModule
from data.datamodule import E2EPerceptionDataModule
from utils.visualization import Visualizer

def parse_args():
    parser = argparse.ArgumentParser(description='Train E2E perception model')
    
    # Data arguments
    parser.add_argument('--train-list', type=str, required=True, help='Path to train clip list')
    parser.add_argument('--val-list', type=str, required=True, help='Path to validation clip list')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--feature-dim', type=int, default=256, help='Feature dimension')
    parser.add_argument('--num-queries', type=int, default=100, help='Number of object queries')
    parser.add_argument('--num-decoder-layers', type=int, default=6, help='Number of decoder layers')
    
    # Training arguments
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--max-epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    
    # Hardware arguments
    parser.add_argument('--accelerator', type=str, default='auto', help='Accelerator to use (auto, gpu, cpu)')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices to use')
    parser.add_argument('--precision', type=str, default='32-true', help='Precision for training')
    
    # Logging arguments
    parser.add_argument('--experiment-name', type=str, default='e2e_perception', help='Name of the experiment')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save outputs')
    
    return parser.parse_args()

def main():
    # Parse arguments
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
    
    # Create data module
    datamodule = E2EPerceptionDataModule(
        train_list=args.train_list,
        val_list=args.val_list,
        camera_ids=camera_ids,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    model = E2EPerceptionModule(
        camera_ids=camera_ids,
        feature_dim=args.feature_dim,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs
    )
    
    # Create loggers
    tb_logger = TensorBoardLogger(
        save_dir=args.save_dir,
        name=args.experiment_name,
        default_hp_metric=False
    )
    csv_logger = CSVLogger(
        save_dir=args.save_dir,
        name=args.experiment_name
    )
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, args.experiment_name, 'checkpoints'),
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val/loss',
        mode='min'
    )
    
    # Create trainer
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        logger=[tb_logger, csv_logger],
        callbacks=[checkpoint_callback],
        gradient_clip_val=0.5,
        accumulate_grad_batches=1,
        deterministic=True
    )
    
    # Train model
    if args.resume:
        trainer.fit(model, datamodule, ckpt_path=args.resume)
    else:
        trainer.fit(model, datamodule)

if __name__ == '__main__':
    main() 