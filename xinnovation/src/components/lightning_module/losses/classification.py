import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Optional
from xinnovation.src.core.registry import LOSSES


@LOSSES.register_module()
class FocalLoss(nn.Module):
    """Focal Loss for classification with class-specific alpha weighting.
    
    Args:
        alpha (float, List[float], or torch.Tensor): Class-specific weighting factor.
            - If float: same weight for all classes
            - If List[float]/torch.Tensor: One weight per class (should match number of classes)
            Defaults to 0.25.
        gamma (float, optional): Focusing parameter. Defaults to 2.0.
        reduction (str, optional): Reduction method. Defaults to 'mean'.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """
    def __init__(
        self,
        alpha: Union[float, List[float]] = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
        loss_weight: float = 1.0,
        pos_weight: List[float] = [1.0]
    ):
        super().__init__()
        
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.register_buffer('pos_weight', torch.tensor(pos_weight, dtype=torch.float32))

        # Register alpha as buffer if it's a list/tensor
        if isinstance(alpha, list):
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
            self.alpha = self.alpha.unsqueeze(0)
        else:
            self.alpha = alpha
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss with batch support.
        
        Args:
            pred (torch.Tensor): Predicted logits
                - Shape: (B, C, *)
                    B: batch size
                    C: number of classes
                    *: any number of additional dimensions
            target (torch.Tensor): Target labels
                - Shape: (B, C, *)
                    B: batch size
                    C: number of classes
                    *: any number of additional dimensions
        Returns:
            torch.Tensor: Computed focal loss
        """
        if isinstance(self.alpha, torch.Tensor):
            assert self.alpha.shape[1] == pred.shape[1], f"Alpha tensor must match number of classes"

        # Sigmoid computation and focal weight
        pred_sigmoid = pred.sigmoid()
        pt = (1 - pred_sigmoid + 1e-8) * target + (pred_sigmoid + 1e-8) * (1 - target)
        focal_weight = pt.pow(self.gamma) * self.alpha
        
        # Compute loss with reduction
        loss = F.binary_cross_entropy_with_logits(
            pred, target, 
            weight=focal_weight.detach(),
            reduction=self.reduction,
            pos_weight=self.pos_weight
        )
        return loss * self.loss_weight

@LOSSES.register_module()
class MultiClassFocalLoss(nn.Module):
    """Multi-class Focal Loss with class-specific alpha weighting.
    
    Args:
        alpha (float, List[float] or torch.Tensor): Per-class weighting factors.
            - If float: Same weight for all classes.
            - If List[float]/torch.Tensor: One weight per class.
        gamma (float, optional): Focusing parameter. Defaults to 2.0.
        reduction (str, optional): Reduction method. Defaults to 'mean'.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """
    
    def __init__(
        self, 
        alpha: Union[float, List[float], torch.Tensor] = 0.25, 
        gamma: float = 2.0, 
        reduction: str = 'mean',
        loss_weight: float = 1.0,
        pos_weight: List[float] = [1.0]
    ):
        super().__init__()

        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.register_buffer('pos_weight', torch.tensor(pos_weight, dtype=torch.float32))
        
        # Register alpha as buffer if it's a list/tensor
        if isinstance(alpha, list):
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
            self.alpha = self.alpha.unsqueeze(0) # (1, C)
        else:
            self.alpha = alpha
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Multi-class Focal Loss.
        
        Args:
            pred (torch.Tensor): Predicted logits
                - Shape: (B, C, *)
                    B: batch size
                    C: number of classes
                    *: any number of additional dimensions
            target (torch.Tensor): Target class indices or one-hot encoded labels
                - Shape: (B, *) for class indices in range [0, C-1]
        Returns:
            torch.Tensor: Computed focal loss
        """
        num_classes = pred.shape[1]
        if isinstance(self.alpha, torch.Tensor):
            assert self.alpha.shape[1] == num_classes, f"Alpha tensor must match number of classes"

        # Ensure target is one-hot encoded
        if target.dim() == 1:
            target = F.one_hot(target, num_classes=num_classes).float()
            # from shape (B, ) to (B, C)

        # Compute softmax probabilities
        prob = F.softmax(pred, dim=1)
        
        # Compute focal weight
        focal_weight = ((1 - prob + 1e-8) * target + (prob + 1e-8) * (1 - target)).pow(self.gamma)
        
        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # Apply focal weight
        focal_weight = focal_weight.detach()
        loss = ce_loss * focal_weight.sum(dim=1)
        
        loss = loss * self.loss_weight
        
        # Apply reduction
        return (
            loss.mean() if self.reduction == 'mean' 
            else loss.sum() if self.reduction == 'sum' 
            else loss
        )