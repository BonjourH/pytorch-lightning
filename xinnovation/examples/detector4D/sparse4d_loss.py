from xinnovation.src.core.registry import LOSSES
from xinnovation.src.components.lightning_module.losses.classification import FocalLoss
from xinnovation.src.components.lightning_module.losses.regression import SmoothL1Loss
import torch.nn as nn
import torch
from dataclasses import dataclass
from typing import List, Dict
from scipy.optimize import linear_sum_assignment
from xinnovation.src.core import TrajParamIndex
from typing import Tuple, Dict, List, Any
from torch.nn import functional as F
import numpy as np
from xinnovation.src.utils.debug_utils import check_nan_or_inf


check_abnormal = False

__all__ = ["Sparse4DLossWithDAC"]


@LOSSES.register_module()
class Sparse4DLossWithDAC(nn.Module):
    def __init__(self, layer_loss_weights: List[float],
                    has_object_loss: dict,
                    attribute_loss: dict,
                    regression_loss: dict,
                    xrel_range: List[float],
                    yrel_range: List[float],
                    use_normalized_motion_cost: bool = False,
                    ct_cost_thresh: float = 10,
                    ct_cls_loss_weight: float = 0.1,
                    **kwargs):
        super().__init__()
        self.xrel_range = xrel_range
        self.yrel_range = yrel_range

        # store matching history of the first batch of different layers
        self.layer_loss_weights = layer_loss_weights
        self.matching_history: Dict[int, List[List[Tuple[np.ndarray, np.ndarray]]]] = {} 
        # layer_idx -> epoch_idx -> List[Tuple[np.ndarray, np.ndarray](hungarian matching results)
        
        self.has_object_loss = LOSSES.build(has_object_loss)
        self.attribute_loss = LOSSES.build(attribute_loss)
        self.regression_loss = LOSSES.build(regression_loss)

        self.use_normalized_motion_cost = use_normalized_motion_cost
        self.ct_cost_thresh = ct_cost_thresh
        self.ct_cls_loss_weight = ct_cls_loss_weight
        self.epoch = 0

    def update_epoch(self, epoch: int):
        self.epoch = epoch
        self.regression_cost_weight = max(0.5, np.exp(-self.epoch / 100) - 0.05)
        self.cls_cost_weight = 1 - self.regression_cost_weight
    
    def reset_matching_history(self):
        self.matching_history = {}

    def get_latest_matching_indices(self, layer_idx: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        if layer_idx not in self.matching_history:
            return []
        return self.matching_history[layer_idx][-1]
    
    @torch.no_grad()
    def _compute_one_to_n_match_results(self, gt_trajs: torch.Tensor, pred_trajs: torch.Tensor, valid_gt_nbs: torch.Tensor) -> torch.Tensor:
        """
        Compute the matching cost matrix between ground truth trajectories and predicted trajectories.
        Args:
            gt_trajs: Ground truth trajectories [B, M, TrajParamIndex.END_OF_INDEX]
            pred_trajs: Predicted trajectories [B, N, TrajParamIndex.END_OF_INDEX]
            valid_gt_nbs: Valid ground truth number of each batch [B]
        Returns:
            matched_pred: Tensor[B_matched, pred_idx]
            matched_gt:   Tensor[B_matched, gt_idx]
            unmatched_pred: Tensor[B_unmatched, pred_idx]
        """
        B, M, N = gt_trajs.shape[0], gt_trajs.shape[1], pred_trajs.shape[1]
        # 1. calculate the cost matrix
        # 1.1 calculate the regression loss, of shape [B, M, N]
        x_cost = torch.cdist(gt_trajs[:, :, TrajParamIndex.X:TrajParamIndex.X+1], pred_trajs[:, :, TrajParamIndex.X:TrajParamIndex.X+1], p=1)
        y_cost = torch.cdist(gt_trajs[:, :, TrajParamIndex.Y:TrajParamIndex.Y+1], pred_trajs[:, :, TrajParamIndex.Y:TrajParamIndex.Y+1], p=1)
        z_cost = torch.cdist(gt_trajs[:, :, TrajParamIndex.Z:TrajParamIndex.Z+1], pred_trajs[:, :, TrajParamIndex.Z:TrajParamIndex.Z+1], p=1)
        dimension_cost = torch.cdist(gt_trajs[:, :, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1], 
                                     pred_trajs[:, :, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1], p=1)
        cost_matrix = x_cost + y_cost + z_cost + dimension_cost
        # 2. set the cost of non-gt to a large value by using valid_gt_nbs
        b_indices = torch.repeat_interleave(torch.arange(B).to(valid_gt_nbs.device), M - valid_gt_nbs)  # (M,)
        m_indices = torch.cat([torch.arange(n, M) for n in valid_gt_nbs]).to(valid_gt_nbs.device)  # (M,)
        cost_matrix[b_indices, m_indices] = 1e6
        # 3. cost_matrix: (B, M, N)，M: 每个 batch 的 gt 数量最大值，N: 每个 batch 的 query 数量
        # 我们现在要求：每个 query（即 dim=-1 的每个位置）在 M 个 gt 中找到最小 cost 的那个
        # min_cost: 每个 query 最小 cost 值
        # matched_gt_idx: 每个 query 对应的 gt 索引（在 M 中）

        # 取最小 cost 对应的 gt 索引
        min_cost, matched_gt_idx = torch.min(cost_matrix, dim=1)  # (B, N)
         # 2. 根据 cost 阈值过滤
        valid_mask = min_cost < self.ct_cost_thresh  # (B, N_pred)

        # 3. 获取 matched 的 batch 和 pred 索引
        matched_batch_idx, matched_pred_idx = torch.where(valid_mask)  # (M,)
        matched_gt_idx = matched_gt_idx[matched_batch_idx, matched_pred_idx]  # (M,)
        
        matched_pred = torch.stack([matched_batch_idx, matched_pred_idx], dim=0)  # (2, M)
        matched_gt = torch.stack([matched_batch_idx, matched_gt_idx], dim=0)      # (2, M)

        # 4. 获取 unmatched 的 prediction 索引
        unmatched_batch_idx, unmatched_pred_idx = torch.where(~valid_mask)
        unmatched_pred = torch.stack([unmatched_batch_idx, unmatched_pred_idx], dim=0)  # (2, M')

        return matched_pred, matched_gt, unmatched_pred
        
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
            indices: Tuple[np.ndarray, np.ndarray] from gt to pred
        """
        # TODO: add 2D BBox GIoU loss
        B, M, N = gt_trajs.shape[0], gt_trajs.shape[1], pred_trajs.shape[1]
        # 1. calculate the cost matrix
        # 1.1 calculate the regression loss, of shape [B, M, N]
        center_cost = torch.cdist(gt_trajs[:, :, TrajParamIndex.X:TrajParamIndex.Z+1], pred_trajs[:, :, TrajParamIndex.X:TrajParamIndex.Z+1], p=1)
        velocity_cost = torch.cdist(gt_trajs[:, :, TrajParamIndex.VX:TrajParamIndex.VY+1], pred_trajs[:, :, TrajParamIndex.VX:TrajParamIndex.VY+1], p=1)
        dimension_cost = torch.cdist(gt_trajs[:, :, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1], 
                                     pred_trajs[:, :, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1], p=1)
        if self.use_normalized_motion_cost:
            center_cost = 1 - torch.exp(-center_cost / 30) # (B, M, N)
            velocity_cost = 0.5 * (1 + torch.tanh(10 * (velocity_cost - 2))) # (B, M, N)
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
        
        cost_matrix = cls_cost * self.cls_cost_weight + regression_cost * self.regression_cost_weight
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
    
    def _save_matching_history(self, layer_idx: int, indices: Tuple[np.ndarray, np.ndarray]):
        # Only record the first batch in each epoch(And in this case we should set shuffle=False in dataloader)
        if layer_idx not in self.matching_history:
            self.matching_history[layer_idx] = []
        if len(indices) > 0:
            self.matching_history[layer_idx].append(indices) # only record the first batch of different layers

    def forward(self, gt_trajs: torch.Tensor, 
                      outputs: List[torch.Tensor], 
                      c_outputs: List[torch.Tensor]=None,
                      step_idx: int = 0, 
                      mode="train") -> Dict[str, torch.Tensor]:
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

        B, M, N = gt_trajs.shape[0], gt_trajs.shape[1], outputs[0].shape[1]

        # 1. Add loss of standard decoders
        for layer_idx, pred_trajs in enumerate(outputs):
            indices = self._compute_hungarian_match_results(gt_trajs, pred_trajs, valid_gt_nbs)
            if mode == "val" and step_idx == 0:
                self._save_matching_history(layer_idx, indices)
            
            # 1.1 create matched mask and reordered gts
            matched_mask = torch.zeros(B, N, dtype=torch.bool, device=pred_trajs.device)
            gt_trajs_reordered = torch.zeros_like(pred_trajs)
            num_positive_preds = 0

            for b, indice in enumerate(indices):
                gt_idx, pred_idx = indice

                gt_trajs_reordered[b, pred_idx] = gt_trajs[b, gt_idx]
                matched_mask[b, pred_idx] = True
                num_positive_preds += len(pred_idx)

            layer_loss_weight = self.layer_loss_weights[layer_idx]

            # 1.2 calculate the has object classification loss for all the preds
            last_idx = TrajParamIndex.END_OF_INDEX if layer_idx > 0 else TrajParamIndex.HAS_OBJECT + 1
            obj_loss = self.has_object_loss(pred_trajs[:, :, TrajParamIndex.HAS_OBJECT:last_idx].flatten(end_dim=1),
                                            gt_trajs_reordered[:, :, TrajParamIndex.HAS_OBJECT:last_idx].flatten(end_dim=1))
            losses[f'layer_{layer_idx}_obj_loss'] = obj_loss * layer_loss_weight
            if num_positive_preds > 0:
                # 1.3 calculate loss for positive preds
                # from [B, N, TrajParamIndex.END_OF_INDEX] to [num_positive_preds, TrajParamIndex.END_OF_INDEX]
                positive_preds = pred_trajs[matched_mask] 
                positive_gts = gt_trajs_reordered[matched_mask]
                if layer_idx == 0:
                    # only apply the loss on positive preds
                    obj_loss = self.has_object_loss(positive_preds[:, TrajParamIndex.HAS_OBJECT],
                                                    positive_gts[:, TrajParamIndex.HAS_OBJECT])
                    losses[f'layer_{layer_idx}_obj_loss'] = obj_loss * layer_loss_weight
                # 1.3.1 calculate the other attribute classification loss for all the positive preds
                # attribute_loss = self.attribute_loss(
                #     positive_preds[:, TrajParamIndex.HAS_OBJECT + 1:],
                #     positive_gts[:, TrajParamIndex.HAS_OBJECT + 1:]
                # ) 
                # losses[f'layer_{layer_idx}_attr_loss'] = attribute_loss * layer_loss_weight
                # 1.3.2 calculate the regression loss for all the positive preds
                regression_loss = self.regression_loss(
                    positive_preds[:, TrajParamIndex.X:TrajParamIndex.HEIGHT + 1],
                    positive_gts[:, TrajParamIndex.X:TrajParamIndex.HEIGHT + 1]
                )
                losses[f'layer_{layer_idx}_reg_loss'] = regression_loss * layer_loss_weight
            
        # 2. Add loss of cross-attention decoder
        if c_outputs is not None:
            for layer_idx, pred_trajs in enumerate(c_outputs):
                weight = self.layer_loss_weights[layer_idx]
                matched_preds_idx, matched_gt_idx, unmatched_preds_idx = self._compute_one_to_n_match_results(gt_trajs, pred_trajs, valid_gt_nbs)
                matched_preds = pred_trajs[matched_preds_idx[0], matched_preds_idx[1]] # [num_matched_preds, TrajParamIndex.END_OF_INDEX]
                matched_gts = gt_trajs[matched_gt_idx[0], matched_gt_idx[1]] # [num_matched_preds, TrajParamIndex.END_OF_INDEX]
                unmatched_preds = pred_trajs[unmatched_preds_idx[0], unmatched_preds_idx[1]] # [num_unmatched_preds, TrajParamIndex.END_OF_INDEX]
                # 2.1 calculate the classification loss for matched preds
                losses[f'layer_{layer_idx}_cls_loss_ct'] = self.has_object_loss(matched_preds[:, TrajParamIndex.HAS_OBJECT],
                                                                            matched_gts[:, TrajParamIndex.HAS_OBJECT]) * weight * self.ct_cls_loss_weight
                # 2.2 calculate the regression loss for matched preds
                losses[f'layer_{layer_idx}_reg_loss_ct'] = self.regression_loss(matched_preds[:, TrajParamIndex.X:TrajParamIndex.HEIGHT + 1],
                                                                            matched_gts[:, TrajParamIndex.X:TrajParamIndex.HEIGHT + 1]) * weight
                # 2.3 calculate the classification loss for unmatched preds
                labels = torch.zeros_like(unmatched_preds[:, TrajParamIndex.HAS_OBJECT])
                losses[f'layer_{layer_idx}_fp_cls_loss_ct'] = self.has_object_loss(unmatched_preds[:, TrajParamIndex.HAS_OBJECT],
                                                                            labels) * weight * self.ct_cls_loss_weight
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
