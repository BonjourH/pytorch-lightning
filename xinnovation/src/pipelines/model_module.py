"""
Base PyTorch Lightning module implementation.
"""
import os
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics import Accuracy, MeanSquaredError
import torchvision.utils

from ..registry import (
    MODEL, BACKBONES, NECKS, HEADS,
    LOSSES, METRICS,
    OPTIMIZERS, SCHEDULERS
)


@MODEL.register_module()
class LightningModelModule(pl.LightningModule):
    """
    Base Lightning Module with enhanced functionality.
    
    This class provides a foundation for building Lightning modules with:
    - Modular component building
    - Automatic metric tracking
    - Flexible loss computation
    - Optimizer and scheduler configuration
    - Visualization support
    """
    
    def __init__(self, hparams: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.build_components()
        self.setup_metrics()
        self.setup_visualization()
        
    def build_components(self):
        """Build model components from config."""
        # Build backbone if specified
        if 'backbone' in self.hparams:
            self.backbone = BACKBONES.build(self.hparams.backbone)
            
        # Build neck if specified
        if 'neck' in self.hparams:
            self.neck = NECKS.build(self.hparams.neck)
            
        # Build head if specified
        if 'head' in self.hparams:
            self.head = HEADS.build(self.hparams.head)
            
        # Build loss functions
        if 'loss' in self.hparams:
            self.loss_fn = LOSSES.build(self.hparams.loss)
            
    def setup_metrics(self):
        """Setup metrics for tracking."""
        self.metrics = {}
        if 'metrics' in self.hparams:
            for name, cfg in self.hparams.metrics.items():
                self.metrics[name] = METRICS.build(cfg)
                
    def setup_visualization(self):
        """Setup visualization components."""
        self.visualization_data = []
        self.max_visualization_samples = self.hparams.get('max_visualization_samples', 8)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of outputs
        """
        outputs = {}
        
        # Extract features from backbone
        if hasattr(self, 'backbone'):
            backbone_feats = self.backbone(x)
            outputs['backbone_feats'] = backbone_feats
            
            # Process through neck if available
            if hasattr(self, 'neck'):
                neck_feats = self.neck(backbone_feats)
                outputs['neck_feats'] = neck_feats
            else:
                neck_feats = backbone_feats
                
            # Generate predictions with head if available
            if hasattr(self, 'head'):
                head_outputs = self.head(neck_feats)
                if isinstance(head_outputs, dict):
                    outputs.update(head_outputs)
                else:
                    outputs['head_outputs'] = head_outputs
                    
        return outputs
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Training step with automatic metric tracking.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Dictionary with loss values
        """
        # Forward pass
        outputs = self(batch['img'])
        
        # Compute losses
        losses = self.compute_losses(outputs, batch)
        
        # Update metrics
        self.update_metrics(outputs, batch, prefix='train')
        
        # Log losses and metrics
        self.log_dict(losses, prefix='train/')
        self.log_metrics(prefix='train/')
        
        return losses
        
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step with metric tracking.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Dictionary with validation metrics
        """
        # Forward pass
        outputs = self(batch['img'])
        
        # Compute losses
        losses = self.compute_losses(outputs, batch)
        
        # Update metrics
        self.update_metrics(outputs, batch, prefix='val')
        
        # Store visualization data
        if batch_idx < self.max_visualization_samples:
            self.store_visualization_data(batch['img'], outputs, batch)
            
        # Log losses and metrics
        self.log_dict(losses, prefix='val/')
        self.log_metrics(prefix='val/')
        
        return losses
        
    def compute_losses(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute losses from outputs and targets.
        
        Args:
            outputs: Model outputs
            batch: Input batch with targets
            
        Returns:
            Dictionary of computed losses
        """
        losses = {}
        if hasattr(self, 'loss_fn'):
            if isinstance(self.loss_fn, dict):
                for name, loss_func in self.loss_fn.items():
                    losses[name] = loss_func(outputs, batch)
            else:
                losses['loss'] = self.loss_fn(outputs, batch)
        return losses
        
    def update_metrics(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], prefix: str = ''):
        """Update metrics with current outputs and targets."""
        for name, metric in self.metrics.items():
            metric(outputs, batch)
            
    def log_metrics(self, prefix: str = ''):
        """Log current metric values."""
        for name, metric in self.metrics.items():
            self.log(f"{prefix}{name}", metric, on_step=False, on_epoch=True)
            
    def store_visualization_data(self, images: torch.Tensor, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        """Store data for visualization."""
        self.visualization_data.append({
            'images': images.detach().cpu(),
            'outputs': {k: v.detach().cpu() for k, v in outputs.items()},
            'targets': {k: v.detach().cpu() for k, v in batch.items() if k != 'img'}
        })
        
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        # Build optimizer
        optimizer_cfg = self.hparams.get('optimizer', {'type': 'AdamW', 'lr': 1e-3})
        optimizer = OPTIMIZERS.build(optimizer_cfg, params=self.parameters())
        
        # Build scheduler if specified
        scheduler_cfg = self.hparams.get('scheduler')
        if scheduler_cfg:
            scheduler = SCHEDULERS.build(scheduler_cfg, optimizer=optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
            
        return optimizer
        
    def on_train_epoch_start(self):
        """Clear visualization data at the start of each training epoch."""
        self.visualization_data = []
        
    def on_validation_epoch_start(self):
        """Clear visualization data at the start of each validation epoch."""
        self.visualization_data = []


