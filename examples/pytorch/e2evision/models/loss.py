import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np

from base import TrajParamIndex

def match_trajectories(
    pred_trajs: torch.Tensor,
    gt_trajs: torch.Tensor,
    frames: int = 10,
    dt: float = 0.1,
    iou_method: str = "iou2"
) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
    """Match predicted trajectories to ground truth using Hungarian algorithm.
    
    Args:
        pred_trajs: Predicted trajectories [B, N, TrajParamIndex.END_OF_INDEX]
        gt_trajs: Ground truth trajectories [B, M, TrajParamIndex.END_OF_INDEX]
        frames: Number of frames to consider for IoU
        dt: Time step between frames
        iou_method: IoU calculation method ("iou" or "iou2")
        
    Returns:
        Tuple containing:
            - List of (pred_idx, gt_idx) pairs
            - Cost matrix [N, M]
    """
    device = pred_trajs.device
    N, M = len(pred_trajs), len(gt_trajs)
    
    if N == 0 or M == 0:
        return [], torch.zeros((N, M), device=device)
    
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
            dist = calculate_trajectory_distance(pred_trajs[i], gt_trajs[j])
            
            # Combined cost
            cost_matrix[i, j] = -iou + 0.1 * dist
    
    # Run Hungarian algorithm
    matches = []
    if N > 0 and M > 0:
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
            matches = list(zip(row_ind, col_ind))
        except ImportError:
            print("Warning: scipy not installed, falling back to greedy matching")
            # Greedy matching
            while len(matches) < min(N, M):
                i, j = torch.argmin(cost_matrix.view(-1)).item() // M, torch.argmin(cost_matrix.view(-1)).item() % M
                if cost_matrix[i, j] < float('inf'):
                    matches.append((i, j))
                    cost_matrix[i, :] = float('inf')
                    cost_matrix[:, j] = float('inf')
                else:
                    break
    
    return matches, cost_matrix

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
            'loss_traj': 1.0,
            'loss_type': 0.1,
            'loss_exist': 1.0
        }
        
        self.frames = frames
        self.dt = dt
        self.iou_method = iou_method
        self.aux_loss_weight = aux_loss_weight
    
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
        for idx, layer_out in enumerate(outputs):
            # Get predictions
            pred_trajs = layer_out['traj_params']  # [B, N, TrajParamIndex.END_OF_INDEX]
            pred_types = layer_out['type_logits']  # [B, N, num_classes]
            
            # Get targets
            gt_trajs = targets['trajs']  # [B, M, TrajParamIndex.END_OF_INDEX]
            
            # Compute losses for this layer
            layer_losses = self._compute_losses(
                pred_trajs=pred_trajs,
                gt_trajs=gt_trajs,
                prefix=f'layer_{idx}_' if idx < num_layers-1 else ''
            )
            
            # Weight auxiliary losses
            if idx < num_layers-1:
                layer_losses = {k: self.aux_loss_weight * v for k, v in layer_losses.items()}
            
            losses.update(layer_losses)
        
        # Weight and sum all losses
        return {k: self.weight_dict[k] * v for k, v in losses.items() if k in self.weight_dict}
    
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
            # Get valid predictions and targets
            valid_preds = pred_trajs[b][pred_trajs[b, :, TrajParamIndex.HAS_OBJECT] > 0.5]
            valid_targets = gt_trajs[b][gt_trajs[b, :, TrajParamIndex.HAS_OBJECT] > 0.5]
            
            # Match trajectories
            matches, cost_matrix = match_trajectories(
                valid_preds,
                valid_targets,
                frames=self.frames,
                dt=self.dt,
                iou_method=self.iou_method
            )
            
            if matches:
                # Trajectory parameter loss
                pred_matched = valid_preds[torch.tensor([i for i, _ in matches], device=device)]
                target_matched = valid_targets[torch.tensor([j for _, j in matches], device=device)]
                
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
                type_loss = F.cross_entropy(
                    pred_matched[:, TrajParamIndex.OBJECT_TYPE].unsqueeze(1),
                    target_matched[:, TrajParamIndex.OBJECT_TYPE].long()
                )
                losses[f'{prefix}loss_type'] = type_loss
            
            # Existence loss
            exist_loss = F.binary_cross_entropy_with_logits(
                pred_trajs[b, :, TrajParamIndex.HAS_OBJECT],
                gt_trajs[b, :, TrajParamIndex.HAS_OBJECT]
            )
            losses[f'{prefix}loss_exist'] = exist_loss
        
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

def calculate_trajectory_distance(traj1: torch.Tensor, traj2: torch.Tensor) -> torch.Tensor:
    """Calculate distance between trajectories.
    
    Args:
        traj1: First trajectory [TrajParamIndex.END_OF_INDEX]
        traj2: Second trajectory [TrajParamIndex.END_OF_INDEX]
        
    Returns:
        Distance value
    """
    pos1 = traj1[[TrajParamIndex.X, TrajParamIndex.Y, TrajParamIndex.Z]]
    pos2 = traj2[[TrajParamIndex.X, TrajParamIndex.Y, TrajParamIndex.Z]]
    return torch.norm(pos1 - pos2) 