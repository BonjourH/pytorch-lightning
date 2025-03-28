import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Any
import lightning.pytorch as pl
from xinnovation.src.core.registry import LIGHTNING_MODULE, DETECTORS, LOSSES
from xinnovation.src.core.builders import build_optimizer, build_scheduler
from easydict import EasyDict as edict


@LIGHTNING_MODULE.register_module()
class LightningDetector(pl.LightningModule):
    def __init__(self, detector: Dict, loss: Dict, optimizer: Dict, scheduler: Dict, **kwargs):
        """Initialize LightningDetector module
        
        Args:
            detector (Dict): Configuration dictionary for detector
            loss (Dict): Configuration dictionary for loss
            optimizer (Dict): Configuration dictionary for optimizer
            scheduler (Dict): Configuration dictionary for scheduler
            **kwargs: Additional arguments
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Create detector network from config
        self.detector = DETECTORS.build(detector)
        
        # Create loss function from config
        self.criterion = LOSSES.build(loss)
        
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers for training
        
        Returns:
            Dict[str, Any]: Dictionary with optimizer and lr_scheduler
        """
        self.optimizer = build_optimizer(self.hparams.optimizer, params=self.parameters())
        self.lr_scheduler = build_scheduler(self.hparams.scheduler, optimizer=self.optimizer)
            
        return { 'optimizer': self.optimizer, 'lr_scheduler': self.lr_scheduler }
    
    def __str__(self):
        """String representation of the model
        
        Returns:
            str: Model information
        """
        info = f"LightningDetector info: detector={self.detector},\n criterion={self.criterion}"
        if hasattr(self, 'optimizer'):
            info += f"\n optimizer={self.optimizer}"
        if hasattr(self, 'lr_scheduler'):
            info += f"\n lr_scheduler={self.lr_scheduler}"
        return info

