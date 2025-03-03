import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import logging
from typing import Dict, List

from base import SourceCameraId
from data import MultiFrameDataset, custom_collate_fn
from model import E2EPerceptionNet
from loss import TrajectoryLoss

def parse_args():
    parser = argparse.ArgumentParser(description='Train E2E perception model')
    
    # Data parameters
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory containing training clips')
    parser.add_argument('--sequence-length', type=int, default=10,
                       help='Number of frames in each sequence')
    
    # Model parameters
    parser.add_argument('--feature-dim', type=int, default=256,
                       help='Feature dimension')
    parser.add_argument('--num-queries', type=int, default=100,
                       help='Number of object queries')
    parser.add_argument('--num-decoder-layers', type=int, default=6,
                       help='Number of decoder layers')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--clip-grad-norm', type=float, default=0.1,
                       help='Gradient clipping norm')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Logging and saving
    parser.add_argument('--log-interval', type=int, default=10,
                       help='How many batches to wait before logging')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default='',
                       help='Resume from checkpoint')
    
    return parser.parse_args()

def setup_logging(save_dir: str):
    """Setup logging configuration."""
    os.makedirs(save_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(save_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )

def save_checkpoint(state: Dict, is_best: bool, save_dir: str):
    """Save checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save latest checkpoint
    checkpoint_path = os.path.join(save_dir, 'checkpoint_latest.pth')
    torch.save(state, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(save_dir, 'checkpoint_best.pth')
        torch.save(state, best_path)

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    args
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    with tqdm(train_loader, desc=f'Epoch {epoch}', leave=False) as pbar:
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            for camera_id in batch['images']:
                batch['images'][camera_id] = batch['images'][camera_id].to(device)
            for camera_id in batch['calibrations']:
                batch['calibrations'][camera_id] = batch['calibrations'][camera_id].to(device)
            batch['ego_states'] = batch['ego_states'].to(device)
            batch['trajs'] = batch['trajs'].to(device)
            
            # Forward pass
            outputs = model(batch)
            
            # Calculate loss
            loss_dict = criterion(outputs, batch)
            loss = loss_dict['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            if batch_idx % args.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
                
                # Log detailed losses
                if batch_idx > 0:
                    log_str = f'Epoch: {epoch}, Batch: {batch_idx}/{num_batches}, Loss: {avg_loss:.4f}'
                    for k, v in loss_dict.items():
                        if k != 'loss':
                            log_str += f', {k}: {v.item():.4f}'
                    logging.info(log_str)
    
    return total_loss / num_batches

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validate', leave=False):
            # Move data to device
            for camera_id in batch['images']:
                batch['images'][camera_id] = batch['images'][camera_id].to(device)
            for camera_id in batch['calibrations']:
                batch['calibrations'][camera_id] = batch['calibrations'][camera_id].to(device)
            batch['ego_states'] = batch['ego_states'].to(device)
            batch['trajs'] = batch['trajs'].to(device)
            
            # Forward pass
            outputs = model(batch)
            
            # Calculate loss
            loss_dict = criterion(outputs, batch)
            total_loss += loss_dict['loss'].item()
    
    return total_loss / len(val_loader)

def main():
    args = parse_args()
    setup_logging(args.save_dir)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
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
    
    # Create datasets
    # TODO: Split data into train and val sets
    train_clips = []  # List of clip directories for training
    val_clips = []    # List of clip directories for validation
    
    train_dataset = MultiFrameDataset(
        clip_dirs=train_clips,
        camera_ids=camera_ids,
        sequence_length=args.sequence_length
    )
    
    val_dataset = MultiFrameDataset(
        clip_dirs=val_clips,
        camera_ids=camera_ids,
        sequence_length=args.sequence_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    # Create model
    model = E2EPerceptionNet(
        camera_ids=camera_ids,
        feature_dim=args.feature_dim,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers
    ).to(device)
    
    # Define loss function
    criterion = TrajectoryLoss().to(device)
    
    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.lr * 0.01
    )
    
    # Initialize training state
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f'Loading checkpoint: {args.resume}')
            checkpoint = torch.load(args.resume, map_location=device)
            
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            
            logging.info(f'Loaded checkpoint from epoch {start_epoch}')
        else:
            logging.error(f'No checkpoint found at: {args.resume}')
            return
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        logging.info(f'\nEpoch {epoch+1}/{args.num_epochs}')
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args)
        logging.info(f'Training Loss: {train_loss:.4f}')
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        logging.info(f'Validation Loss: {val_loss:.4f}')
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'args': args
        }, is_best, args.save_dir)
        
        if is_best:
            logging.info(f'New best model with validation loss: {val_loss:.4f}')

if __name__ == '__main__':
    main() 