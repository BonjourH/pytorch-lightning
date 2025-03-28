from xinnovation.src.core.registry import LOSSES
import torch.nn as nn
import torch

__all__ = ["Sparse4DLoss"]

@LOSSES.register_module()
class Sparse4DLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, pred, target):
        return 0