import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Any
from xinnovation.src.core.registry import DETECTORS
from easydict import EasyDict as edict

__all__ = ["Sparse4DDetector"]

@DETECTORS.register_module()
class Sparse4DDetector(nn.Module):
    def __init__(self, backbone: Dict, head: Dict, **kwargs):
        super().__init__()
        # self.backbone = build_backbone(backbone)
        # self.head = build_head(head)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x