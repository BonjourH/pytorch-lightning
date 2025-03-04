import lightning as L
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from base import SourceCameraId
from models.network import E2EPerceptionNet
from models.loss import TrajectoryLoss

class E2EPerceptionModule(L.LightningModule):
    """Lightning module for end-to-end perception."""
    
    def __init__(self,
                 camera_ids: List[SourceCameraId],
                 feature_dim: int = 256,
                 num_queries: int = 100,
                 num_decoder_layers: int = 6,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 max_epochs: int = 100):
        super().__init__()
        self.save_hyperparameters(ignore=['camera_ids'])
        
        # Create network
        self.net = E2EPerceptionNet(
            camera_ids=camera_ids,
            feature_dim=feature_dim,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers
        )
        
        # Create loss function
        self.criterion = TrajectoryLoss()
        
        # Initialize validation metrics
        self.val_step_outputs = []
    
    def forward(self, batch: Dict) -> List[Dict]:
        return self.net(batch)
    
    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Training step."""
        # Forward pass
        outputs = self(batch)
        
        # Compute loss
        loss_dict = self.criterion(outputs, batch)
        
        # Log losses
        for name, value in loss_dict.items():
            self.log(f"train/{name}", value, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss_dict
    
    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Validation step."""
        # Forward pass
        outputs = self(batch)
        
        # Compute loss
        loss_dict = self.criterion(outputs, batch)
        
        # Store outputs for epoch end
        self.val_step_outputs.append({
            'loss_dict': loss_dict,
            'outputs': outputs[-1],  # Only keep final predictions
            'targets': batch
        })
        
        # Log losses
        for name, value in loss_dict.items():
            self.log(f"val/{name}", value, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss_dict
    
    def on_validation_epoch_end(self):
        """Compute validation metrics at epoch end."""
        if not self.val_step_outputs:
            return
        
        # Aggregate predictions and targets
        all_preds = []
        all_targets = []
        
        for output in self.val_step_outputs:
            pred_trajs = output['outputs']['traj_params']
            pred_types = output['outputs']['type_logits'].argmax(dim=-1)
            gt_trajs = output['targets']['trajs']
            
            # Filter valid predictions and targets
            valid_preds = pred_trajs[pred_trajs[..., 11] > 0.5]  # HAS_OBJECT flag
            valid_targets = gt_trajs[gt_trajs[..., 11] > 0.5]
            
            all_preds.extend(valid_preds)
            all_targets.extend(valid_targets)
        
        # Compute metrics
        metrics = self._compute_metrics(all_preds, all_targets)
        
        # Log metrics
        for name, value in metrics.items():
            self.log(f"val/{name}", value, on_epoch=True)
        
        # Clear outputs
        self.val_step_outputs.clear()
    
    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Test step."""
        return self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch: Dict, batch_idx: int, dataloader_idx: int = 0) -> Dict:
        """Prediction step."""
        outputs = self(batch)
        return outputs[-1]  # Return final predictions
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Create optimizer
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Create scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1
            }
        }
    
    def _compute_metrics(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]) -> Dict:
        """Compute evaluation metrics."""
        metrics = {}
        
        # TODO: Implement detailed metrics computation
        # - Position error
        # - Velocity error
        # - Classification accuracy
        # - Detection metrics (precision, recall, F1)
        
        return metrics 