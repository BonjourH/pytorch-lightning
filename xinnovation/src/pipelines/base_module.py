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

from e2e_perception.src.utils.registry import MODEL, LOSSES


@LIGHTNING_MODULE.register()
class BaseModule(pl.LightningModule):
    """
    Base Lightning Module for perception tasks.
    
    Args:
        backbone_cfg: Configuration for the backbone
        neck_cfg: Configuration for the neck (optional)
        head_cfg: Configuration for the head
        loss_cfg: Configuration for losses
        optimizer_cfg: Configuration for optimizer
        lr_scheduler_cfg: Configuration for learning rate scheduler
        data_cfg: Configuration for datasets
        model_cfg: Additional model configuration
    """
    
    def __init__(
        self,
        backbone_cfg: Dict[str, Any],
        neck_cfg: Optional[Dict[str, Any]] = None,
        head_cfg: Dict[str, Any] = None,
        loss_cfg: Dict[str, Any] = None,
        optimizer_cfg: Dict[str, Any] = None,
        lr_scheduler_cfg: Optional[Dict[str, Any]] = None,
        data_cfg: Dict[str, Any] = None,
        model_cfg: Dict[str, Any] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model components
        self.backbone = self._build_backbone(backbone_cfg)
        self.neck = self._build_neck(neck_cfg) if neck_cfg else None
        self.head = self._build_head(head_cfg) if head_cfg else None
        
        # Initialize loss functions
        self.loss_fn = self._build_losses(loss_cfg) if loss_cfg else None
        
        # Initialize other parameters
        self.optimizer_cfg = optimizer_cfg
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        
        # Visualization data
        self.visualization_data = []
        
    def _build_backbone(self, cfg: Dict[str, Any]) -> nn.Module:
        """
        Build backbone network from config.
        
        Args:
            cfg: Backbone configuration
            
        Returns:
            Initialized backbone module
        """
        backbone_type = cfg.pop('type')
        backbone = MODEL.get(backbone_type)(**cfg)
        return backbone
    
    def _build_neck(self, cfg: Dict[str, Any]) -> nn.Module:
        """
        Build neck network from config.
        
        Args:
            cfg: Neck configuration
            
        Returns:
            Initialized neck module
        """
        neck_type = cfg.pop('type')
        neck = MODEL.get(neck_type)(**cfg)
        return neck
    
    def _build_head(self, cfg: Dict[str, Any]) -> nn.Module:
        """
        Build head network from config.
        
        Args:
            cfg: Head configuration
            
        Returns:
            Initialized head module
        """
        head_type = cfg.pop('type')
        head = MODEL.get(head_type)(**cfg)
        return head
    
    def _build_losses(self, cfg: Dict[str, Any]) -> Dict[str, nn.Module]:
        """
        Build loss functions from config.
        
        Args:
            cfg: Loss configuration
            
        Returns:
            Dictionary of loss functions
        """
        losses = {}
        for loss_name, loss_cfg in cfg.items():
            loss_type = loss_cfg.pop('type')
            loss_fn = LOSSES.get(loss_type)(**loss_cfg)
            losses[loss_name] = loss_fn
        return losses
    
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of outputs from model components
        """
        outputs = {}
        
        # Extract features from backbone
        backbone_feats = self.backbone(x)
        outputs['backbone_feats'] = backbone_feats
        
        # Process features through neck if available
        if self.neck:
            neck_feats = self.neck(backbone_feats)
            outputs['neck_feats'] = neck_feats
        else:
            neck_feats = backbone_feats
        
        # Generate predictions with head if available
        if self.head:
            if hasattr(self.head, 'forward'):
                head_outputs = self.head(neck_feats)
                # Handle different return types from head
                if isinstance(head_outputs, dict):
                    outputs.update(head_outputs)
                elif isinstance(head_outputs, tuple) and len(head_outputs) == 2:
                    # For DetectionHead that returns (cls_scores, bbox_preds)
                    outputs['cls_scores'] = head_outputs[0]
                    outputs['bbox_preds'] = head_outputs[1]
                else:
                    # Just store the output directly
                    outputs['head_outputs'] = head_outputs
        
        return outputs
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Training step for Lightning.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Dictionary with loss values
        """
        # Extract inputs and targets
        imgs = batch['img']
        targets = {k: v for k, v in batch.items() if k != 'img'}
        
        # Forward pass
        outputs = self(imgs)
        
        # Compute losses
        losses = self._compute_losses(outputs, targets)
        
        # Log losses
        for loss_name, loss_value in losses.items():
            self.log(f'train/{loss_name}', loss_value, prog_bar=True)
        
        return {'loss': losses['loss']}
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step for Lightning.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Dictionary with loss values
        """
        # Extract inputs and targets
        imgs = batch['img']
        targets = {k: v for k, v in batch.items() if k != 'img'}
        
        # Forward pass
        outputs = self(imgs)
        
        # Compute losses
        losses = self._compute_losses(outputs, targets)
        
        # Store some results for visualization
        if batch_idx < 5:  # Store only a few batches for visualization
            self._store_visualization_data(imgs, outputs, targets)
        
        # Log losses
        for loss_name, loss_value in losses.items():
            self.log(f'val/{loss_name}', loss_value, prog_bar=True)
        
        return {'val_loss': losses['loss']}
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step for Lightning.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Dictionary with metrics
        """
        # Similar to validation step but for test data
        return self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Prediction step for Lightning.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Dictionary with predictions
        """
        # Extract inputs
        imgs = batch['img']
        
        # Forward pass
        outputs = self(imgs)
        
        # Extract predictions
        predictions = self._get_predictions(outputs)
        
        return predictions
    
    def _compute_losses(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute losses from outputs and targets.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            
        Returns:
            Dictionary of computed losses
        """
        # This is a placeholder. In a real implementation, this would compute actual losses
        # based on the model outputs and targets
        
        # For demonstration, return a dummy loss
        # Make sure we have at least one tensor to create a dummy loss
        tensor_keys = [k for k in outputs if isinstance(outputs[k], torch.Tensor)]
        
        if tensor_keys:
            # Create a dummy loss from existing tensors
            dummy_loss = sum([outputs[k].sum() * 0 for k in tensor_keys])
        else:
            # If no tensors found, create a new one
            dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return {'loss': dummy_loss, 'dummy_loss': dummy_loss}
    
    def _get_predictions(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract predictions from model outputs.
        
        Args:
            outputs: Model outputs
            
        Returns:
            Dictionary of predictions
        """
        # This is a placeholder. In a real implementation, this would extract
        # useful predictions from the model outputs
        
        # For demonstration, just return the outputs as predictions
        return outputs
    
    def _store_visualization_data(self, imgs: torch.Tensor, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> None:
        """
        Store data for visualization.
        
        Args:
            imgs: Input images
            outputs: Model outputs
            targets: Ground truth targets
        """
        # This is a placeholder. In a real implementation, this would store
        # visualization data for use in callbacks
        
        # For demonstration, store some sample data
        for i in range(min(2, imgs.size(0))):  # Store only a couple of examples
            self.visualization_data.append({
                'image': imgs[i].detach().cpu(),
                'bboxes': targets.get('gt_bboxes', [None])[i].detach().cpu() if 'gt_bboxes' in targets else None,
                'labels': targets.get('gt_labels', [None])[i].detach().cpu() if 'gt_labels' in targets else None,
                'scores': torch.ones(1).detach().cpu()  # Dummy scores
            })
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        # Default optimization configuration
        if not self.optimizer_cfg:
            optimizer = optim.Adam(self.parameters(), lr=1e-3)
        else:
            # Get optimizer type and parameters
            optimizer_type = self.optimizer_cfg.pop('type', 'Adam')
            optimizer_params = self.optimizer_cfg
            
            # Initialize optimizer
            optimizer_cls = getattr(optim, optimizer_type)
            optimizer = optimizer_cls(self.parameters(), **optimizer_params)
        
        # If no scheduler is specified, just return the optimizer
        if not self.lr_scheduler_cfg:
            return optimizer
        
        # Get scheduler type and parameters
        scheduler_type = self.lr_scheduler_cfg.pop('type', 'StepLR')
        scheduler_params = self.lr_scheduler_cfg
        
        # Initialize scheduler
        scheduler_cls = getattr(optim.lr_scheduler, scheduler_type)
        scheduler = scheduler_cls(optimizer, **scheduler_params)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def on_train_epoch_start(self) -> None:
        """Clear visualization data at the start of each training epoch."""
        self.visualization_data = []
    
    def on_validation_epoch_start(self) -> None:
        """Clear visualization data at the start of each validation epoch."""
        self.visualization_data = [] 