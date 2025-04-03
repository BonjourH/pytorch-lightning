from xinnovation.src.core.registry import LOSSES
import torch.nn as nn
import torch
from dataclasses import dataclass
from typing import List, Dict
from scipy.optimize import linear_sum_assignment
from xinnovation.src.core import TrajParamIndex
from typing import Tuple, Dict, List, Any
from torch.nn import functional as F
import numpy as np


__all__ = ["Sparse4DLossWithDAC"]


@LOSSES.register_module()
class Sparse4DLossWithDAC(nn.Module):
    def __init__(self, layer_loss_weights: List[float], **kwargs):
        super().__init__()
        # store matching history of the first batch of different layers
        self.matching_history: Dict[int, List[Tuple[int, int]]] = {} # layer_idx -> List[(gt_idx, pred_idx)]
    
    @torch.no_grad()
    def _compute_hungarian_match_results(self, gt_trajs: torch.Tensor, pred_trajs: torch.Tensor, valid_gt_nbs: torch.Tensor) -> torch.Tensor:
        """
        Compute the matching cost matrix between ground truth trajectories and predicted trajectories.
        
        Args:
            gt_trajs: Ground truth trajectories [B, M, TrajParamIndex.END_OF_INDEX]
            pred_trajs: Predicted trajectories [B, N, TrajParamIndex.END_OF_INDEX]
            valid_gt_nbs: Valid ground truth number of each batch [B]
        Intermediate_results:
            cls_cost: Classification cost matrix [B, M, N]
            center_cost: Center loss matrix [B, M, N]
            cost_matrix: Matching cost matrix [B, M, N]
        
        Returns:
            indices: List[Tuple[int, int]] where each tuple is (gt_idx, pred_idx)
        """
        # TODO: add 2D BBox GIoU loss
        B, M, N = gt_trajs.shape[0], gt_trajs.shape[1], pred_trajs.shape[1]
        # 1. calculate the cost matrix
        # 1.1 calculate the regression loss, of shape [B, M, N]
        center_cost = torch.cdist(gt_trajs[:, :, TrajParamIndex.X:TrajParamIndex.Z+1], pred_trajs[:, :, TrajParamIndex.X:TrajParamIndex.Z+1], p=1)
        center_cost = 1 - torch.exp(-center_cost / 30) # (B, M, N)
        velocity_cost = torch.cdist(gt_trajs[:, :, TrajParamIndex.VX:TrajParamIndex.VY+1], pred_trajs[:, :, TrajParamIndex.VX:TrajParamIndex.VY+1], p=2)
        velocity_cost = 0.5 * (1 + torch.tanh(10 * (velocity_cost - 2))) # (B, M, N)
        dimension_cost = torch.cdist(gt_trajs[:, :, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1], pred_trajs[:, :, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1], p=2)
        dimension_cost = 0.5 * (1 + torch.tanh(10 * (dimension_cost - 2))) # (B, M, N)
        # giou_loss_matrix = 0 # TODO: define the giou loss in 4D space(two trajectories)
        regression_cost = center_cost * 0.7 + velocity_cost * 0.2 + dimension_cost * 0.1

        # NOTE: cost function choice
        # 1. center_cost use `1 - exp(-center_cost / 30)`, which is sensitive to small center distance but not sensitive to large center distance
        # 2. velocity_cost use `0.5 * (1 + tanh(10 * (velocity_cost - 2)))`, which is sensitive to small velocity difference but not sensitive to large velocity difference
        # 3. dimension_cost use `0.5 * (1 + tanh(10 * (dimension_cost - 2)))`, which is sensitive to small dimension difference but not sensitive to large dimension difference

        # 1.2 calculate the classification loss(only consider the attribute HAS_OBJECT)
        cls_cost = compute_has_object_cls_cost_matrix(gt_trajs, pred_trajs) # (B, M, N)
        # 1.4 calculate the composite loss
        cost_matrix = cls_cost * 0.5 + regression_cost * 0.5
        # 2. calculate the indices
        indices = []
        for b in range(B):
            origin_indice = linear_sum_assignment(cost_matrix[b].detach().cpu().numpy())
            # 2.1 remove the fp matches (M trajs is the pre-defined number of gt trajs, however, the real number of gt trajs is less than M)
            real_gt_count = valid_gt_nbs[b].item()
            gt_idx, pred_idx = origin_indice
            # 只保留前k个匹配对
            k = min(real_gt_count, len(gt_idx))
            gt_idx = gt_idx[:k]
            pred_idx = pred_idx[:k]
            indices.append((gt_idx, pred_idx))
        return indices
    
    def _save_matching_history(self, layer_idx: int, indices: List[Tuple[int, int]]):
        if layer_idx not in self.matching_history:
            self.matching_history[layer_idx] = []
        if len(indices) > 0:
            self.matching_history[layer_idx].append(indices[0]) # only record the first batch of different layers

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
        # calculate the valid gt number of each batch
        trajs_mask = gt_trajs[..., TrajParamIndex.HAS_OBJECT] == 1.0
        valid_gt_nbs = trajs_mask.sum(dim=1) # [B]
        
        losses = {}

        B, N, M = gt_trajs.shape[0], gt_trajs.shape[1], outputs[0].shape[1]

        # 1. Add loss of standard decoders
        for layer_idx, pred_trajs in enumerate(outputs):
            indices = self._compute_hungarian_match_results(gt_trajs, pred_trajs, valid_gt_nbs)
            self._save_matching_history(layer_idx, indices)

            # 收集所有batch中匹配的预测和真实轨迹
            matched_preds = []
            matched_gts = []
            
            # 创建mask来标识未匹配的预测轨迹
            unmatched_mask = torch.ones(B, N, dtype=torch.bool, device=pred_trajs.device)
            
            # 收集匹配的轨迹和标记未匹配的轨迹
            for b, indice in enumerate(indices):
                gt_idx, pred_idx = indice
                # 收集匹配的轨迹
                matched_preds.append(pred_trajs[b, pred_idx])
                matched_gts.append(gt_trajs[b, gt_idx])
                
                # 标记已匹配的轨迹
                unmatched_mask[b, pred_idx] = False
            
            layer_loss_weight = self.hyper_parameters.layer_loss_weights[layer_idx]
            # 堆叠所有匹配的轨迹
            if matched_preds:  # 如果存在匹配的轨迹
                matched_preds = torch.cat(matched_preds, dim=0)  # [total_matches, TrajParamIndex.END_OF_INDEX]
                matched_gts = torch.cat(matched_gts, dim=0)      # [total_matches, TrajParamIndex.END_OF_INDEX]
                
                # 计算匹配轨迹的losses (true positives)
                # 分类loss
                cls_loss = F.binary_cross_entropy_with_logits(
                    matched_preds[:, TrajParamIndex.HAS_OBJECT],
                    matched_gts[:, TrajParamIndex.HAS_OBJECT]
                )
                losses[f'layer_{layer_idx}_cls_loss'] = cls_loss * layer_loss_weight
                
                # 回归loss
                reg_loss = F.smooth_l1_loss(
                    matched_preds[:, TrajParamIndex.X:TrajParamIndex.END_OF_INDEX],
                    matched_gts[:, TrajParamIndex.X:TrajParamIndex.END_OF_INDEX]
                )
                losses[f'layer_{layer_idx}_reg_loss'] = reg_loss * layer_loss_weight
            
            # 获取所有未匹配的预测轨迹
            unmatched_preds = pred_trajs[unmatched_mask]  # [total_unmatched, TrajParamIndex.END_OF_INDEX]
            
            # 计算未匹配轨迹的losses (false positives)
            if len(unmatched_preds) > 0:
                fp_cls_loss = F.binary_cross_entropy_with_logits(
                    unmatched_preds[:, TrajParamIndex.HAS_OBJECT],
                    torch.zeros_like(unmatched_preds[:, TrajParamIndex.HAS_OBJECT])
                )
                losses[f'layer_{layer_idx}_fp_cls_loss'] = fp_cls_loss * layer_loss_weight
        
        # 2. Add loss of cross-attention decoder
        if c_outputs is not None:
            for layer_idx, pred_trajs in enumerate(c_outputs[1:]):
                # TODO: calculate the loss
                pass
        
        losses['loss'] = sum(losses.values())
        return losses

def compute_has_object_cls_cost_matrix(gt_trajs: torch.Tensor, pred_trajs: torch.Tensor) -> torch.Tensor:
    """Compute classification cost matrix between ground truth and predictions using Focal Loss.
    
    Args:
        gt_trajs: Ground truth trajectories [B, M, TrajParamIndex.END_OF_INDEX]
        pred_trajs: Predicted trajectories [B, N, TrajParamIndex.END_OF_INDEX]
        
    Returns:
        cls_cost: Classification cost matrix [B, M, N]
    """
    B, M, _ = gt_trajs.shape
    _, N, _ = pred_trajs.shape
    
    # Get objectness scores for ground truth and predictions
    gt_obj = gt_trajs[..., TrajParamIndex.HAS_OBJECT]    # [B, M]
    pred_obj = pred_trajs[..., TrajParamIndex.HAS_OBJECT]  # [B, N]
    
    # Expand dimensions to create cost matrix
    gt_obj = gt_obj.unsqueeze(-1).expand(-1, -1, N)    # [B, M, N]
    pred_obj = pred_obj.unsqueeze(1).expand(-1, M, -1)  # [B, M, N]
    
    # Focal Loss parameters
    alpha = 0.25  # alpha parameter for focal loss
    gamma = 2.0   # gamma parameter for focal loss
    
    # Convert logits to probabilities
    pred_prob = torch.sigmoid(pred_obj)
    
    # Compute focal loss components
    neg_cost = (1 - alpha) * (pred_prob ** gamma) * (-(1 - pred_prob + 1e-8).log())
    pos_cost = alpha * ((1 - pred_prob) ** gamma) * (-(pred_prob + 1e-8).log())
    
    # Combine positive and negative costs based on ground truth
    cls_cost = torch.where(gt_obj == 1.0, pos_cost, neg_cost)  # [B, M, N]
    
    return cls_cost
