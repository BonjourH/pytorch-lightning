from xinnovation.src.core.registry import LOSSES
import torch.nn as nn
import torch
from dataclasses import dataclass
from typing import List, Dict

__all__ = ["Sparse4DLoss"]


@LOSSES.register_module()
class ObstacleStateLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, pred, target):
        return 0
    
# # LIGHTNING (顶层)
# └── LIGHTNING_MODULE
#     └── LOSSES             # 损失函数 (FocalLoss, SmoothL1, DiceLoss)
#         ├── OBSTACLE_STATE_PREDICTION_LOSS
#         │   ├── POSITION_LOSS
#         │   │   ├── L1_NORM
#         │   │   └── L2_NORM
#         │   ├── VELOCITY_LOSS
#         │   │   ├── ABSOLUTE_VELOCITY_LOSS
#         │   │   └── RELATIVE_VELOCITY_LOSS
#         │   ├── DIMENSION_LOSS
#         │   │   ├── GEOMETRIC_LOSS
#         │   │   └── SIZE_CONSISTENCY_LOSS
#         │   ├── TYPE_CLASSIFICATION_LOSS
#         │   │   ├── CROSS_ENTROPY
#         │   │   └── FOCAL_LOSS
#         │   └── COMPOSITE_LOSS
#         │       ├── WEIGHTED_AGGREGATION
#         │       └── UNCERTAINTY_AWARE_BALANCING

@dataclass
class Sparse4DLossConfig:
    layer_loss_weights: List[float]


@LOSSES.register_module()
class Sparse4DLoss(nn.Module):
    def __init__(self, layer_loss_weights: List[float], **kwargs):
        super().__init__()
        
    def forward(self, pred, target):
        return 0