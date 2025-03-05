import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from base import TrajParamIndex

def match_trajectories(
    pred_trajs: torch.Tensor,
    gt_trajs: torch.Tensor,
    frames: int = 10,
    dt: float = 0.1,
    iou_method: str = "iou2"
) -> Tuple[List[Tuple[int, int]], List[int], torch.Tensor]:
    """Match predicted trajectories to ground truth using Hungarian algorithm.
    
    Args:
        pred_trajs: Predicted trajectories [B, N, TrajParamIndex.END_OF_INDEX]
        gt_trajs: Ground truth trajectories [B, M, TrajParamIndex.END_OF_INDEX]
        frames: Number of frames to consider for IoU
        dt: Time step between frames
        iou_method: IoU calculation method ("iou" or "iou2")
        
    Returns:
        Tuple containing:
            - List of (pred_idx, gt_idx) pairs for matched trajectories
            - List of pred_idx for unmatched predictions
            - Cost matrix [N, M]
    """
    device = pred_trajs.device
    N, M = len(pred_trajs), len(gt_trajs)
    
    if M == 0:
        return [], list(range(N)), torch.zeros((N, M), device=device)
    
    # Compute cost matrix
    cost_matrix = torch.zeros((N, M), device=device)
    
    for i in range(N):
        for j in range(M):
            # IoU cost
            if iou_method == "iou2":
                iou = calculate_trajectory_bev_iou2(
                    pred_trajs[i:i+1],
                    gt_trajs[j:j+1],
                    torch.arange(frames, device=device) * dt
                )
            else:
                iou = calculate_trajectory_bev_iou(
                    pred_trajs[i:i+1],
                    gt_trajs[j:j+1],
                    torch.arange(frames, device=device) * dt
                )
            
            # Distance cost
            dist_score = calculate_trajectory_distance_score(pred_trajs[i], gt_trajs[j])
            
            score = iou * 0.6 + dist_score * 0.4
            
            if pred_trajs[i, TrajParamIndex.HAS_OBJECT] < 0.5:
                score *= 0.5
            
            obj_index = int(torch.argmax(pred_trajs[i][TrajParamIndex.CAR:]))
            if gt_trajs[j, obj_index] != 1:
                score *= 0.5
            
            # Combined cost
            cost_matrix[i, j] = 1 - score
    
    # Run Hungarian algorithm
    matches = []
    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
    matches = list(zip(row_ind, col_ind))
    
    # Find unmatched prediction indices
    matched_pred_indices = set(i for i, _ in matches)
    unmatched_pred_indices = [i for i in range(N) if i not in matched_pred_indices]
    
    return matches, unmatched_pred_indices, cost_matrix

class TrajectoryLoss(nn.Module):
    """Loss function for trajectory prediction."""
    
    def __init__(self, 
                 weight_dict=None,
                 frames: int = 10,
                 dt: float = 0.1,
                 iou_method: str = "iou2",
                 aux_loss_weight: float = 0.5):
        super().__init__()
        
        # Set loss weights
        self.weight_dict = weight_dict or {
            'loss_pos': 1.0,      # 位置损失权重
            'loss_vel': 1.0,      # 速度损失权重
            'loss_acc': 1.0,      # 加速度损失权重
            'loss_dim': 1.0,      # 尺寸损失权重
            'loss_yaw': 1.0,      # 偏航角损失权重
            'loss_type': 0.1,     # 类型损失权重
            'loss_attr': 0.1,     # 属性损失权重
            'fp_loss_exist': 1.0, # 假阳性存在损失权重
        }
        
        self.layer_loss_weights = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
        
        self.frames = frames
        self.dt = dt
        self.iou_method = iou_method
    
    def forward(self, outputs: List[Dict], targets: Dict) -> Dict:
        """Forward pass.
        
        Args:
            outputs: List of decoder outputs at each layer
            targets: Ground truth
            
        Returns:
            Dictionary of losses
        """
        # Initialize losses
        losses = {}
        
        # Process each decoder layer
        num_layers = len(outputs)
        for idx, pred_trajs in enumerate(outputs):
            # [B, N, TrajParamIndex.END_OF_INDEX]
            
            # Get targets
            gt_trajs = targets['trajs']  # [B, M, TrajParamIndex.END_OF_INDEX]
            
            # Compute losses for this layer
            layer_losses = self._compute_losses(
                pred_trajs=pred_trajs,
                gt_trajs=gt_trajs,
                prefix=f'layer_{idx}_'
            )
            
            losses.update(layer_losses)
        
        # Weight and sum all losses
        weighted_losses = {}
        for k, v in losses.items():
            layer_idx = int(k.split('_')[1])
            post_fix = "_".join(k.split('_')[2:])
            weight = self.weight_dict[post_fix] * self.layer_loss_weights[layer_idx]
            weighted_losses[k] = weight * v
        
        # Calculate total loss
        total_loss = sum(weighted_losses.values())
        weighted_losses['loss'] = total_loss
        
        return weighted_losses
    
    def _compute_losses(self, 
                       pred_trajs: torch.Tensor, 
                       gt_trajs: torch.Tensor,
                       prefix: str = '') -> Dict[str, torch.Tensor]:
        """Compute losses for a single decoder layer.
        
        Args:
            pred_trajs: Predicted trajectories [B, N, TrajParamIndex.END_OF_INDEX]
            gt_trajs: Ground truth trajectories [B, M, TrajParamIndex.END_OF_INDEX]
            prefix: Prefix for loss names
            
        Returns:
            Dictionary of losses
        """
        B = pred_trajs.shape[0]
        device = pred_trajs.device
        losses = {}
        
        # Process each batch
        for b in range(B):
            valid_targets = gt_trajs[b][gt_trajs[b, :, TrajParamIndex.HAS_OBJECT] > 0.5]
            
            # Match trajectories
            matches, unmatched_pred_indices, cost_matrix = match_trajectories(
                pred_trajs[b],
                valid_targets,
                frames=self.frames,
                dt=self.dt,
                iou_method=self.iou_method
            )
            
            # For the unmatched pred traj, we only penalize the existence loss
            if unmatched_pred_indices:
                pred_unmatched = pred_trajs[b][torch.tensor(unmatched_pred_indices, device=device)]
                exist_loss = F.binary_cross_entropy_with_logits(
                    pred_unmatched[:, TrajParamIndex.HAS_OBJECT],
                    torch.zeros_like(pred_unmatched[:, TrajParamIndex.HAS_OBJECT])
                )
                losses[f'{prefix}fp_loss_exist'] = exist_loss
            
            # For the matched pred traj, we add the loss for all the predictions
            if matches:
                pred_matched = pred_trajs[b][torch.tensor(list(zip(*matches))[0], device=device)]
                target_matched = valid_targets[torch.tensor(list(zip(*matches))[1], device=device)]
                # Trajectory parameter loss
                # Position loss
                pos_loss = F.l1_loss(
                    pred_matched[:, [TrajParamIndex.X, TrajParamIndex.Y, TrajParamIndex.Z]],
                    target_matched[:, [TrajParamIndex.X, TrajParamIndex.Y, TrajParamIndex.Z]]
                )
                losses[f'{prefix}loss_pos'] = pos_loss
                
                # Velocity loss
                vel_loss = F.l1_loss(
                    pred_matched[:, [TrajParamIndex.VX, TrajParamIndex.VY]],
                    target_matched[:, [TrajParamIndex.VX, TrajParamIndex.VY]]
                )
                losses[f'{prefix}loss_vel'] = vel_loss
                
                # Acceleration loss
                acc_loss = F.l1_loss(
                    pred_matched[:, [TrajParamIndex.AX, TrajParamIndex.AY]],
                    target_matched[:, [TrajParamIndex.AX, TrajParamIndex.AY]]
                )
                losses[f'{prefix}loss_acc'] = acc_loss
                
                # Dimension loss
                dim_loss = F.l1_loss(
                    pred_matched[:, [TrajParamIndex.LENGTH, TrajParamIndex.WIDTH, TrajParamIndex.HEIGHT]],
                    target_matched[:, [TrajParamIndex.LENGTH, TrajParamIndex.WIDTH, TrajParamIndex.HEIGHT]]
                )
                losses[f'{prefix}loss_dim'] = dim_loss
                
                # Yaw loss
                yaw_loss = F.l1_loss(
                    pred_matched[:, TrajParamIndex.YAW],
                    target_matched[:, TrajParamIndex.YAW]
                )
                losses[f'{prefix}loss_yaw'] = yaw_loss
                
                # Type loss
                type_loss = F.binary_cross_entropy_with_logits(
                    pred_matched[:, TrajParamIndex.CAR:TrajParamIndex.UNKNOWN+1],
                    target_matched[:, TrajParamIndex.CAR:TrajParamIndex.UNKNOWN+1]
                )
                losses[f'{prefix}loss_type'] = type_loss
                
                # Attribute loss
                attr_loss = F.binary_cross_entropy_with_logits(
                    pred_matched[:, TrajParamIndex.HAS_OBJECT:TrajParamIndex.OCCLUDED+1],
                    target_matched[:, TrajParamIndex.HAS_OBJECT:TrajParamIndex.OCCLUDED+1]
                )
                losses[f'{prefix}loss_attr'] = attr_loss
        
        # Average losses across batch
        return {k: v.mean() for k, v in losses.items()}

def calculate_trajectory_bev_iou(traj1: torch.Tensor, traj2: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """Calculate bird's eye view IoU between trajectories.
    
    Args:
        traj1: First trajectory [B, TrajParamIndex.END_OF_INDEX]
        traj2: Second trajectory [B, TrajParamIndex.END_OF_INDEX]
        frames: Frame timestamps [T]
        
    Returns:
        IoU value
    """
    # TODO: Implement BEV IoU calculation
    return torch.zeros(1, device=traj1.device)

def calculate_trajectory_bev_iou2(traj1: torch.Tensor, traj2: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """Calculate bird's eye view IoU between trajectories using improved method.
    
    Args:
        traj1: First trajectory [B, TrajParamIndex.END_OF_INDEX]
        traj2: Second trajectory [B, TrajParamIndex.END_OF_INDEX]
        frames: Frame timestamps [T]
        
    Returns:
        IoU value
    """
    # TODO: Implement improved BEV IoU calculation
    return torch.zeros(1, device=traj1.device)

def calculate_trajectory_distance_score(traj1: torch.Tensor, traj2: torch.Tensor) -> torch.Tensor:
    """Calculate distance between trajectories.
    
    Args:
        traj1: First trajectory [TrajParamIndex.END_OF_INDEX]
        traj2: Second trajectory [TrajParamIndex.END_OF_INDEX]
        
    Returns:
        Distance value
    """
    # calculate distance between two points
    pos1 = traj1[[TrajParamIndex.X, TrajParamIndex.Y, TrajParamIndex.Z]]
    pos2 = traj2[[TrajParamIndex.X, TrajParamIndex.Y, TrajParamIndex.Z]]
    dist = torch.norm(pos1 - pos2)
    
    # Use average size of both objects to normalize distance
    size1 = traj1[[TrajParamIndex.LENGTH, TrajParamIndex.WIDTH, TrajParamIndex.HEIGHT]]
    size2 = traj2[[TrajParamIndex.LENGTH, TrajParamIndex.WIDTH, TrajParamIndex.HEIGHT]]
    size_diff = torch.norm(size1 - size2)
    
    
    # Normalized distance score that decays with distance relative to object size
    dist_score = torch.exp(-(dist + size_diff) / (size_diff + 1e-6))
    return dist_score
