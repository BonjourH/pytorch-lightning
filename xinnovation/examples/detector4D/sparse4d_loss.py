from xinnovation.src.core.registry import LOSSES
import torch.nn as nn
import torch
from dataclasses import dataclass
from typing import List, Dict
from scipy.optimize import linear_sum_assignment
from xinnovation.src.core import TrajParamIndex
from typing import Tuple, Dict, List, Any
from torch.nn import functional as F


__all__ = ["Sparse4DLossWithDAC"]


@LOSSES.register_module()
class Sparse4DLossWithDAC(nn.Module):
    def __init__(self, layer_loss_weights: List[float], **kwargs):
        super().__init__()
        self.obstacle_present_threshold = 0.5
        # store matching history of the first batch of different layers
        self.matching_history: Dict[int, List[Tuple[int, int]]] = {} # layer_idx -> List[(pred_idx, gt_idx)]
    
    @torch.no_grad()
    def _compute_hungarian_match_results(self, pred_trajs: torch.Tensor, gt_trajs: torch.Tensor) -> torch.Tensor:
        """
        Compute the matching cost matrix between predicted trajectories and ground truth trajectories.
        
        Args:
            pred_trajs: Predicted trajectories [B, N, TrajParamIndex.END_OF_INDEX]
            gt_trajs: Ground truth trajectories [B, M, TrajParamIndex.END_OF_INDEX]
            
        Intermediate_results:
            cls_cost: Classification cost matrix [B, N, M]
            center_cost: Center loss matrix [B, N, M]
            cost_matrix: Matching cost matrix [B, N, M]
        
        Returns:
            indices: List[Tuple[int, int]]
        """
        # TODO: add 2D BBox GIoU loss
        B, N, M = pred_trajs.shape[0], pred_trajs.shape[1], gt_trajs.shape[1]
        # 1. calculate the cost matrix
        # 1.1 calculate the cregression loss, of shape [B, N, M]
        center_cost = torch.cdist(pred_trajs[:, :, TrajParamIndex.X:TrajParamIndex.Z+1], gt_trajs[:, :, TrajParamIndex.X:TrajParamIndex.Z+1], p=1)
        center_cost = 1 - torch.exp(-center_cost / 30) # (B, N, M)
        velocity_cost = torch.cdist(pred_trajs[:, :, TrajParamIndex.VX:TrajParamIndex.VY+1], gt_trajs[:, :, TrajParamIndex.VX:TrajParamIndex.VY+1], p=2)
        velocity_cost = 0.5 * (1 + torch.tanh(10 * (velocity_cost - 2))) # (B, N, M)
        dimension_cost = torch.cdist(pred_trajs[:, :, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1], gt_trajs[:, :, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1], p=2)
        dimension_cost = 0.5 * (1 + torch.tanh(10 * (dimension_cost - 2))) # (B, N, M)
        # giou_loss_matrix = 0 # TODO: define the giou loss in 4D space(two trajectories)
        regression_cost = center_cost * 0.7 + velocity_cost * 0.2 + dimension_cost * 0.1

        # NOTE: cost function choice
        # 1. center_cost use `1 - exp(-center_cost / 30)`, which is sensitive to small center distance but not sensitive to large center distance
        # 2. velocity_cost use `0.5 * (1 + tanh(10 * (velocity_cost - 2)))`, which is sensitive to small velocity difference but not sensitive to large velocity difference
        # 3. dimension_cost use `0.5 * (1 + tanh(10 * (dimension_cost - 2)))`, which is sensitive to small dimension difference but not sensitive to large dimension difference

        # 1.2 calculate the classification loss(only consider the attribute HAS_OBJECT)
        cls_cost = compute_has_object_cls_cost_matrix(pred_trajs, gt_trajs) # (B, N, M)
        # 1.4 calculate the composite loss
        cost_matrix = cls_cost * 0.5 + regression_cost * 0.5
        # 2. calculate the indices
        indices = []
        for b in range(B):
            indices.append(linear_sum_assignment(cost_matrix[b].detach().cpu().numpy()))
        return indices

    def forward(self, gt_trajs: torch.Tensor, outputs: List[torch.Tensor], c_outputs: List[torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute losses.
        
        Args:
            gt_trajs: Ground truth trajectories [B, M, TrajParamIndex.END_OF_INDEX]
            outputs: List of predicted trajectories [B, N, TrajParamIndex.END_OF_INDEX] of decoders from each layer
            c_outputs: List of predicted trajectories from cross-attention decoder [B, N, TrajParamIndex.END_OF_INDEX]
        
        Returns:
            losses: Dictionary of losses
        """
        losses = {}

        # 1. Add loss of standard decoders
        for idx, pred_trajs in enumerate(outputs):
            if idx not in self.matching_history:
                self.matching_history[idx] = []
            indices = self._compute_hungarian_match_results(pred_trajs, gt_trajs)
            self.matching_history[idx].append(indices[0]) # only record the first batch of different layers
            for indice in indices:
                # TODO: calculate the loss
                # layer_losses = self._compute_losses_with_indices(
                #     pred_trajs=pred_trajs_b,
                #     gt_trajs=gt_trajs_b,
                #     indices=indices,
                #     prefix=f'layer_{idx}_'
                # )
                # losses.update(layer_losses)
                pass
 
        # 2. Add loss of cross-attention decoder
        if c_outputs is not None:
            for idx, pred_trajs in enumerate(c_outputs[1:]):
                # TODO: calculate the loss
                # layer_losses = self._compute_losses_one_gt_match_n_preds(
                #     pred_trajs=pred_trajs,
                #     gt_trajs=gt_trajs,
                #     prefix=f'cross_layer_{idx}_',
                #     layer_idx=idx
                # )
                # losses.update(layer_losses)
                pass
        
        losses['loss'] = sum(losses.values())
        return losses

def compute_has_object_cls_cost_matrix(pred_trajs: torch.Tensor, gt_trajs: torch.Tensor) -> torch.Tensor:
    """Compute classification cost matrix between predictions and ground truth using Focal Loss.
    
    Args:
        pred_trajs: Predicted trajectories [B, N, TrajParamIndex.END_OF_INDEX]
        gt_trajs: Ground truth trajectories [B, M, TrajParamIndex.END_OF_INDEX]
        
    Returns:
        cls_cost: Classification cost matrix [B, N, M]
    """
    B, N, _ = pred_trajs.shape
    _, M, _ = gt_trajs.shape
    
    # Get objectness scores for predictions and ground truth
    pred_obj = pred_trajs[..., TrajParamIndex.HAS_OBJECT]  # [B, N]
    gt_obj = gt_trajs[..., TrajParamIndex.HAS_OBJECT]     # [B, M]
    
    # Expand dimensions to create cost matrix
    pred_obj = pred_obj.unsqueeze(-1).expand(-1, -1, M)  # [B, N, M]
    gt_obj = gt_obj.unsqueeze(1).expand(-1, N, -1)       # [B, N, M]
    
    # Focal Loss parameters
    alpha = 0.25  # alpha parameter for focal loss
    gamma = 2.0   # gamma parameter for focal loss
    
    # Convert logits to probabilities
    pred_prob = torch.sigmoid(pred_obj)
    
    # Compute focal loss components
    neg_cost = (1 - alpha) * (pred_prob ** gamma) * (-(1 - pred_prob + 1e-8).log())
    pos_cost = alpha * ((1 - pred_prob) ** gamma) * (-(pred_prob + 1e-8).log())
    
    # Combine positive and negative costs based on ground truth
    cls_cost = torch.where(gt_obj > 0.5, pos_cost, neg_cost)  # [B, N, M]
    
    return cls_cost
