import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Optional
from xinnovation.src.core.registry import LOSSES


@LOSSES.register_module()
class SmoothL1Loss(nn.Module):
    """Smooth L1 Loss (Huber Loss) for bounding box regression.
    
    Args:
        beta (float): Threshold parameter that determines the smooth transition
            point from L1 to L2 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Options: 'none', 'mean', 'sum'. Default: 'mean'.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """
    def __init__(
        self,
        beta: float = 1.0,
        reduction: str = 'mean',
        loss_weight: float = 1.0
    ):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward function of SmoothL1Loss.
        
        Args:
            pred (torch.Tensor): Predictions of bbox regression.
                Shape: (B, 4) or (B, N, 4).
            target (torch.Tensor): Target of bbox regression.
                Shape: (B, 4) or (B, N, 4).
                
        Returns:
            torch.Tensor: Calculated loss
        """
        assert pred.size() == target.size()
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.beta, 
                          0.5 * diff * diff / self.beta,
                          diff - 0.5 * self.beta)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss * self.loss_weight


@LOSSES.register_module()
class IoULoss(nn.Module):
    """IoU Loss for bounding box regression.
    
    Args:
        mode (str): Loss mode, options are 'iou', 'giou', 'diou', 'ciou'.
            Default: 'giou'.
        reduction (str): Specifies the reduction to apply.
            Options: 'none', 'mean', 'sum'. Default: 'mean'.
        loss_weight (float): Weight of loss. Default: 1.0.
    """
    def __init__(
        self,
        mode: str = 'giou',
        reduction: str = 'mean',
        loss_weight: float = 1.0
    ):
        super().__init__()
        assert mode in ['iou', 'giou', 'diou', 'ciou'], f"Mode {mode} is not supported"
        self.mode = mode
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward function.
        
        Args:
            pred (torch.Tensor): Predicted bboxes, in format [x1, y1, x2, y2].
                Shape: (B, 4) or (B, N, 4).
            target (torch.Tensor): Target bboxes, in format [x1, y1, x2, y2].
                Shape: (B, 4) or (B, N, 4).
                
        Returns:
            torch.Tensor: Calculated loss
        """
        assert pred.size() == target.size()
        
        # Extract coordinates
        pred_x1, pred_y1, pred_x2, pred_y2 = pred.unbind(dim=-1)
        target_x1, target_y1, target_x2, target_y2 = target.unbind(dim=-1)
        
        # Calculate areas
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        
        # Calculate intersection
        x1_max = torch.maximum(pred_x1, target_x1)
        y1_max = torch.maximum(pred_y1, target_y1)
        x2_min = torch.minimum(pred_x2, target_x2)
        y2_min = torch.minimum(pred_y2, target_y2)
        
        w_intersect = torch.clamp(x2_min - x1_max, min=0)
        h_intersect = torch.clamp(y2_min - y1_max, min=0)
        intersection = w_intersect * h_intersect
        
        # Calculate union
        union = pred_area + target_area - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-7)
        
        if self.mode == 'iou':
            loss = 1 - iou
        elif self.mode == 'giou':
            # Calculate enclosing box
            x1_min = torch.minimum(pred_x1, target_x1)
            y1_min = torch.minimum(pred_y1, target_y1)
            x2_max = torch.maximum(pred_x2, target_x2)
            y2_max = torch.maximum(pred_y2, target_y2)
            
            enclosing_area = (x2_max - x1_min) * (y2_max - y1_min)
            giou = iou - (enclosing_area - union) / (enclosing_area + 1e-7)
            loss = 1 - giou
        elif self.mode == 'diou' or self.mode == 'ciou':
            # Calculate center points
            pred_cx = (pred_x1 + pred_x2) / 2
            pred_cy = (pred_y1 + pred_y2) / 2
            target_cx = (target_x1 + target_x2) / 2
            target_cy = (target_y1 + target_y2) / 2
            
            # Calculate central point distance
            central_point_distance = torch.sqrt(
                (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2 + 1e-7)
            
            # Calculate diagonal distance of enclosing box
            x1_min = torch.minimum(pred_x1, target_x1)
            y1_min = torch.minimum(pred_y1, target_y1)
            x2_max = torch.maximum(pred_x2, target_x2)
            y2_max = torch.maximum(pred_y2, target_y2)
            
            enclosing_diagonal = torch.sqrt(
                (x2_max - x1_min) ** 2 + (y2_max - y1_min) ** 2 + 1e-7)
            
            diou = iou - central_point_distance ** 2 / (enclosing_diagonal ** 2 + 1e-7)
            
            if self.mode == 'ciou':
                # Calculate aspect ratio consistency term
                pred_w = pred_x2 - pred_x1
                pred_h = pred_y2 - pred_y1
                target_w = target_x2 - target_x1
                target_h = target_y2 - target_y1
                
                v = (4 / (torch.pi ** 2)) * torch.pow(
                    torch.atan(target_w / torch.clamp(target_h, min=1e-7)) - 
                    torch.atan(pred_w / torch.clamp(pred_h, min=1e-7)), 2)
                
                with torch.no_grad():
                    alpha = v / (1 - iou + v + 1e-7)
                
                ciou = diou - alpha * v
                loss = 1 - ciou
            else:
                loss = 1 - diou
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss * self.loss_weight
