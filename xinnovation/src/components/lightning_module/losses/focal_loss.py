import torch
import torch.nn as nn
import torch.nn.functional as F
from xinnovation.src.core.registry import LOSSES

@LOSSES.register_module()
class FocalLoss(nn.Module):
    """Focal Loss for dense object detection.
    
    Args:
        alpha (float): Alpha parameter for focal loss
        gamma (float): Gamma parameter for focal loss
        reduction (str): Reduction method for loss
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred, target):
        """Forward pass.
        
        Args:
            pred (torch.Tensor): Predicted logits
            target (torch.Tensor): Target labels
            
        Returns:
            torch.Tensor: Computed focal loss
        """
        pred_sigmoid = pred.sigmoid()
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(pred, target, weight=focal_weight)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
            
        return loss 