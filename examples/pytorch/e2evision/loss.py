import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple
import numpy as np
from base import Trajectory, TrajParamIndex

class TrajectoryMatcher:
    """Match predicted trajectories to ground truth using Hungarian algorithm."""
    def __init__(self, cost_weights: Dict[str, float] = None):
        self.cost_weights = cost_weights or {
            'center': 1.0,
            'velocity': 1.0,
            'acceleration': 0.5,
            'yaw': 1.0,
            'size': 1.0
        }
    
    def __call__(self, 
                 pred_trajs: List[Trajectory],
                 gt_trajs: List[Trajectory],
                 timestamp: float) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
        """Match predicted trajectories to ground truth."""
        N = len(pred_trajs)
        M = len(gt_trajs)
        device = next(iter(pred_trajs[0].__dict__.values())).device
        
        cost_matrix = torch.zeros(N, M, device=device)
        
        for i, pred in enumerate(pred_trajs):
            for j, gt in enumerate(gt_trajs):
                # Center position cost
                pred_center = torch.from_numpy(pred.center(timestamp))
                gt_center = torch.from_numpy(gt.center(timestamp))
                center_cost = torch.norm(pred_center - gt_center, p=1)
                
                # Velocity cost
                pred_vel = torch.from_numpy(pred.velocity_at(timestamp))
                gt_vel = torch.from_numpy(gt.velocity_at(timestamp))
                vel_cost = torch.norm(pred_vel - gt_vel, p=1)
                
                # Acceleration cost
                acc_cost = torch.norm(
                    torch.from_numpy(pred.acceleration - gt.acceleration),
                    p=1
                )
                
                # Yaw cost
                yaw_diff = abs(pred.yaw_at(timestamp) - gt.yaw_at(timestamp))
                yaw_cost = min(yaw_diff, 2*np.pi - yaw_diff)
                
                # Size cost
                size_cost = torch.norm(
                    torch.from_numpy(pred.dimensions - gt.dimensions),
                    p=1
                )
                
                cost_matrix[i, j] = (
                    self.cost_weights['center'] * center_cost +
                    self.cost_weights['velocity'] * vel_cost +
                    self.cost_weights['acceleration'] * acc_cost +
                    self.cost_weights['yaw'] * yaw_cost +
                    self.cost_weights['size'] * size_cost
                )
        
        # Run Hungarian algorithm
        pred_idx, gt_idx = linear_sum_assignment(cost_matrix.cpu().numpy())
        indices = [(int(i), int(j)) for i, j in zip(pred_idx, gt_idx)]
        
        return indices, cost_matrix

class TrajectoryLoss(nn.Module):
    """Loss function for trajectory prediction."""
    def __init__(self, matcher: TrajectoryMatcher):
        super().__init__()
        self.matcher = matcher
        
    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss between predicted and ground truth trajectories.
        Args:
            outputs: Dict containing:
                - trajectories: [B, N, 9] final predictions
                - aux_trajectories: List[[B, N, 9]] auxiliary predictions
            targets: Dict containing:
                - trajectories: [B, M, 9] ground truth
                - valid: [B, M] valid mask
        Returns:
            Dict of losses
        """
        pred_trajs = outputs['trajectories']
        gt_trajs = targets['trajectories']
        gt_valid = targets['valid']
        
        # Match predictions to ground truth
        indices, cost_matrix = self.matcher(pred_trajs, gt_trajs, 0.0)
        
        # Initialize losses
        losses = {
            'center': 0.0,
            'velocity': 0.0,
            'acceleration': 0.0,
            'yaw': 0.0,
            'size': 0.0
        }
        
        # Compute losses for each batch
        B = len(indices)
        for b, b_indices in enumerate(indices):
            if len(b_indices) == 0:  # No matches in this batch
                continue
                
            pred_idx = b_indices[:, 0]
            gt_idx = b_indices[:, 1]
            
            # Get matched pairs
            pred_matched = pred_trajs[b, pred_idx]
            gt_matched = gt_trajs[b, gt_idx]
            
            # Center position loss (L1)
            losses['center'] += F.l1_loss(
                pred_matched[..., TrajParamIndex.X:TrajParamIndex.Z+1],
                gt_matched[..., TrajParamIndex.X:TrajParamIndex.Z+1],
                reduction='mean'
            )
            
            # Velocity loss (L1)
            losses['velocity'] += F.l1_loss(
                pred_matched[..., TrajParamIndex.VX:TrajParamIndex.VY+1],
                gt_matched[..., TrajParamIndex.VX:TrajParamIndex.VY+1],
                reduction='mean'
            )
            
            # Acceleration loss (L1)
            losses['acceleration'] += F.l1_loss(
                pred_matched[..., TrajParamIndex.AX:TrajParamIndex.AY+1],
                gt_matched[..., TrajParamIndex.AX:TrajParamIndex.AY+1],
                reduction='mean'
            )
            
            # Yaw loss (circular)
            yaw_diff = torch.abs(
                pred_matched[..., TrajParamIndex.YAW] - 
                gt_matched[..., TrajParamIndex.YAW]
            )
            yaw_diff = torch.min(yaw_diff, 2*torch.pi - yaw_diff)
            losses['yaw'] += yaw_diff.mean()
            
            # Size loss (L1)
            losses['size'] += F.l1_loss(
                pred_matched[..., TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1],
                gt_matched[..., TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1],
                reduction='mean'
            )
        
        # Average over batches
        for k in losses:
            losses[k] = losses[k] / B
            
        # Add auxiliary losses if available
        if 'aux_trajectories' in outputs:
            aux_weight = 0.5
            for aux_trajs in outputs['aux_trajectories']:
                aux_indices, _ = self.matcher(aux_trajs, gt_trajs, 0.0)
                aux_losses = self.compute_losses(aux_trajs, gt_trajs, aux_indices)
                for k in aux_losses:
                    losses[f'aux_{k}'] = aux_weight * aux_losses[k]
                aux_weight *= 0.5  # Reduce weight for later auxiliary predictions
        
        # Total loss
        losses['total'] = sum(v for k, v in losses.items() if 'aux_' not in k)
        
        return losses

    def compute_losses(self,
                      pred_trajs: torch.Tensor,
                      gt_trajs: torch.Tensor,
                      indices: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Helper function to compute individual losses."""
        losses = {
            'center': 0.0,
            'velocity': 0.0,
            'acceleration': 0.0,
            'yaw': 0.0,
            'size': 0.0
        }
        
        B = len(indices)
        for b, b_indices in enumerate(indices):
            if len(b_indices) == 0:
                continue
                
            pred_idx = b_indices[:, 0]
            gt_idx = b_indices[:, 1]
            
            pred_matched = pred_trajs[b, pred_idx]
            gt_matched = gt_trajs[b, gt_idx]
            
            # Center position loss (L1)
            losses['center'] += F.l1_loss(
                pred_matched[..., TrajParamIndex.X:TrajParamIndex.Z+1],
                gt_matched[..., TrajParamIndex.X:TrajParamIndex.Z+1],
                reduction='mean'
            )
            
            # Velocity loss (L1)
            losses['velocity'] += F.l1_loss(
                pred_matched[..., TrajParamIndex.VX:TrajParamIndex.VY+1],
                gt_matched[..., TrajParamIndex.VX:TrajParamIndex.VY+1],
                reduction='mean'
            )
            
            # Acceleration loss (L1)
            losses['acceleration'] += F.l1_loss(
                pred_matched[..., TrajParamIndex.AX:TrajParamIndex.AY+1],
                gt_matched[..., TrajParamIndex.AX:TrajParamIndex.AY+1],
                reduction='mean'
            )
            
            # Yaw loss (circular)
            yaw_diff = torch.abs(
                pred_matched[..., TrajParamIndex.YAW] - 
                gt_matched[..., TrajParamIndex.YAW]
            )
            yaw_diff = torch.min(yaw_diff, 2*torch.pi - yaw_diff)
            losses['yaw'] += yaw_diff.mean()
            
            # Size loss (L1)
            losses['size'] += F.l1_loss(
                pred_matched[..., TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1],
                gt_matched[..., TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1],
                reduction='mean'
            )
            
        for k in losses:
            losses[k] = losses[k] / B
            
        return losses 