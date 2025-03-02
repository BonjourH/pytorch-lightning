import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional
import numpy as np
from base import (
    TrajParamIndex, AttributeType, ObjectType,
    ObstacleTrajectory, Point3DAccMotion
)

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
                 pred_trajs: torch.Tensor,
                 gt_trajs: torch.Tensor) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
        """Match predicted trajectories to ground truth.
        
        Args:
            pred_trajs: Tensor[N, TrajParamIndex.END_OF_INDEX] - Predicted trajectories
            gt_trajs: Tensor[M, TrajParamIndex.END_OF_INDEX] - Ground truth trajectories
            
        Returns:
            Tuple containing:
            - List of (pred_idx, gt_idx) pairs
            - Cost matrix used for matching
        """
        N, M = pred_trajs.shape[0], gt_trajs.shape[0]
        device = pred_trajs.device
        
        if N == 0 or M == 0:
            return [], torch.zeros(0, 0, device=device)
        
        cost_matrix = torch.zeros(N, M, device=device)
        
        for i in range(N):
            for j in range(M):
                pred = pred_trajs[i]
                gt = gt_trajs[j]
                
                # Center position cost (L1 norm)
                center_cost = torch.abs(pred[TrajParamIndex.X:TrajParamIndex.Z+1] - 
                                        gt[TrajParamIndex.X:TrajParamIndex.Z+1]).sum()
                
                # Velocity cost (L1 norm)
                vel_cost = torch.abs(pred[TrajParamIndex.VX:TrajParamIndex.VY+1] - 
                                     gt[TrajParamIndex.VX:TrajParamIndex.VY+1]).sum()
                
                # Acceleration cost (L1 norm)
                acc_cost = torch.abs(pred[TrajParamIndex.AX:TrajParamIndex.AY+1] - 
                                    gt[TrajParamIndex.AX:TrajParamIndex.AY+1]).sum()
                
                # Yaw cost (circular)
                yaw_diff = torch.abs(pred[TrajParamIndex.YAW] - gt[TrajParamIndex.YAW])
                yaw_cost = torch.min(yaw_diff, 2*torch.pi - yaw_diff)
                
                # Size cost (L1 norm)
                size_cost = torch.abs(pred[TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1] - 
                                     gt[TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1]).sum()
                
                # Weighted sum
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
                - predicted_trajs: Tensor[B, N, TrajParamIndex.END_OF_INDEX] - final predictions
                - aux_trajs: List[Tensor[B, N, TrajParamIndex.END_OF_INDEX]] - auxiliary predictions
            targets: Dict containing:
                - gt_trajs: Tensor[B, M, TrajParamIndex.END_OF_INDEX] - ground truth
                - gt_valid: Tensor[B, M] - valid mask for ground truth
        Returns:
            Dict of losses
        """
        pred_trajs = outputs['predicted_trajs']  # [B, N, TrajParamIndex.END_OF_INDEX]
        gt_trajs = targets['gt_trajs']  # [B, M, TrajParamIndex.END_OF_INDEX]
        gt_valid = targets.get('gt_valid')  # [B, M]
        
        B = pred_trajs.shape[0]
        device = pred_trajs.device
        
        # Initialize losses
        losses = {
            'center': torch.tensor(0.0, device=device),
            'velocity': torch.tensor(0.0, device=device),
            'acceleration': torch.tensor(0.0, device=device),
            'yaw': torch.tensor(0.0, device=device),
            'size': torch.tensor(0.0, device=device),
            'object_type': torch.tensor(0.0, device=device),
            'attributes': torch.tensor(0.0, device=device)
        }
        
        # Filter valid ground truth for each batch and compute losses
        for b in range(B):
            # Get batch predictions
            b_pred = pred_trajs[b]  # [N, TrajParamIndex.END_OF_INDEX]
            
            # Filter valid predictions (those with HAS_OBJECT > 0.5)
            pred_valid = b_pred[:, TrajParamIndex.HAS_OBJECT] > 0.5
            b_pred_valid = b_pred[pred_valid]  # [valid_N, TrajParamIndex.END_OF_INDEX]
            
            # Get batch ground truth
            b_gt = gt_trajs[b]  # [M, TrajParamIndex.END_OF_INDEX]
            
            # Filter valid ground truth if gt_valid is provided
            if gt_valid is not None:
                b_gt_valid = b_gt[gt_valid[b]]  # [valid_M, TrajParamIndex.END_OF_INDEX]
            else:
                # Otherwise use HAS_OBJECT flag to filter valid ground truth
                gt_valid_mask = b_gt[:, TrajParamIndex.HAS_OBJECT] > 0.5
                b_gt_valid = b_gt[gt_valid_mask]  # [valid_M, TrajParamIndex.END_OF_INDEX]
            
            # Skip if either valid predictions or ground truth is empty
            if len(b_pred_valid) == 0 or len(b_gt_valid) == 0:
                continue
            
            # Match predictions to ground truth
            indices, _ = self.matcher(b_pred_valid, b_gt_valid)
            
            if len(indices) == 0:
                continue
            
            # Extract matched pairs
            pred_indices, gt_indices = zip(*indices)
            pred_matched = b_pred_valid[list(pred_indices)]
            gt_matched = b_gt_valid[list(gt_indices)]
            
            # Center position loss (L1)
            losses['center'] += F.l1_loss(
                pred_matched[:, TrajParamIndex.X:TrajParamIndex.Z+1],
                gt_matched[:, TrajParamIndex.X:TrajParamIndex.Z+1],
                reduction='mean'
            )
            
            # Velocity loss (L1)
            losses['velocity'] += F.l1_loss(
                pred_matched[:, TrajParamIndex.VX:TrajParamIndex.VY+1],
                gt_matched[:, TrajParamIndex.VX:TrajParamIndex.VY+1],
                reduction='mean'
            )
            
            # Acceleration loss (L1)
            losses['acceleration'] += F.l1_loss(
                pred_matched[:, TrajParamIndex.AX:TrajParamIndex.AY+1],
                gt_matched[:, TrajParamIndex.AX:TrajParamIndex.AY+1],
                reduction='mean'
            )
            
            # Yaw loss (circular)
            yaw_diff = torch.abs(
                pred_matched[:, TrajParamIndex.YAW] - 
                gt_matched[:, TrajParamIndex.YAW]
            )
            yaw_diff = torch.min(yaw_diff, 2*torch.pi - yaw_diff)
            losses['yaw'] += yaw_diff.mean()
            
            # Size loss (L1)
            losses['size'] += F.l1_loss(
                pred_matched[:, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1],
                gt_matched[:, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1],
                reduction='mean'
            )
            
            # Object type loss (cross-entropy)
            # Convert object type to integer index
            pred_type = pred_matched[:, TrajParamIndex.OBJECT_TYPE]
            gt_type = gt_matched[:, TrajParamIndex.OBJECT_TYPE].long()
            losses['object_type'] += F.cross_entropy(
                pred_type.unsqueeze(1),
                gt_type
            )
            
            # Attributes loss (binary cross-entropy)
            attr_indices = [TrajParamIndex.HAS_OBJECT, TrajParamIndex.STATIC, TrajParamIndex.OCCLUDED]
            pred_attrs = pred_matched[:, attr_indices]
            gt_attrs = gt_matched[:, attr_indices]
            losses['attributes'] += F.binary_cross_entropy_with_logits(
                pred_attrs,
                gt_attrs,
                reduction='mean'
            )
        
        # Average over batches
        valid_batch_count = 0
        for b in range(B):
            # Check if there are valid ground truth in this batch
            if gt_valid is not None:
                if gt_valid[b].sum() > 0:
                    valid_batch_count += 1
            else:
                if (gt_trajs[b, :, TrajParamIndex.HAS_OBJECT] > 0.5).sum() > 0:
                    valid_batch_count += 1
                    
        # Avoid division by zero
        valid_batch_count = max(1, valid_batch_count)
        
        for k in losses:
            losses[k] = losses[k] / valid_batch_count
            
        # Add auxiliary losses if available
        if 'aux_trajs' in outputs:
            aux_weight = 0.5
            for aux_idx, aux_trajs in enumerate(outputs['aux_trajs']):
                aux_losses = self.compute_auxiliary_losses(
                    aux_trajs, gt_trajs, gt_valid
                )
                
                for k, v in aux_losses.items():
                    losses[f'aux{aux_idx}_{k}'] = aux_weight * v
                
                aux_weight *= 0.5  # Reduce weight for later auxiliary predictions
        
        # Total loss
        losses['total'] = sum(v for k, v in losses.items() if not k.startswith('aux'))
        
        return losses
        
    def compute_auxiliary_losses(self,
                               aux_trajs: torch.Tensor,
                               gt_trajs: torch.Tensor,
                               gt_valid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute losses for auxiliary trajectory predictions."""
        B = aux_trajs.shape[0]
        device = aux_trajs.device
        
        # Initialize losses
        losses = {
            'center': torch.tensor(0.0, device=device),
            'velocity': torch.tensor(0.0, device=device),
            'acceleration': torch.tensor(0.0, device=device),
            'yaw': torch.tensor(0.0, device=device),
            'size': torch.tensor(0.0, device=device),
            'object_type': torch.tensor(0.0, device=device),
            'attributes': torch.tensor(0.0, device=device)
        }
        
        # Filter valid ground truth for each batch and compute losses
        for b in range(B):
            # Get batch predictions
            b_pred = aux_trajs[b]  # [N, TrajParamIndex.END_OF_INDEX]
            
            # Filter valid predictions (those with HAS_OBJECT > 0.5)
            pred_valid = b_pred[:, TrajParamIndex.HAS_OBJECT] > 0.5
            b_pred_valid = b_pred[pred_valid]  # [valid_N, TrajParamIndex.END_OF_INDEX]
            
            # Get batch ground truth
            b_gt = gt_trajs[b]  # [M, TrajParamIndex.END_OF_INDEX]
            
            # Filter valid ground truth if gt_valid is provided
            if gt_valid is not None:
                b_gt_valid = b_gt[gt_valid[b]]  # [valid_M, TrajParamIndex.END_OF_INDEX]
            else:
                # Otherwise use HAS_OBJECT flag to filter valid ground truth
                gt_valid_mask = b_gt[:, TrajParamIndex.HAS_OBJECT] > 0.5
                b_gt_valid = b_gt[gt_valid_mask]  # [valid_M, TrajParamIndex.END_OF_INDEX]
            
            # Skip if either valid predictions or ground truth is empty
            if len(b_pred_valid) == 0 or len(b_gt_valid) == 0:
                continue
            
            # Match predictions to ground truth
            indices, _ = self.matcher(b_pred_valid, b_gt_valid)
            
            if len(indices) == 0:
                continue
            
            # Extract matched pairs
            pred_indices, gt_indices = zip(*indices)
            pred_matched = b_pred_valid[list(pred_indices)]
            gt_matched = b_gt_valid[list(gt_indices)]
            
            # Center position loss (L1)
            losses['center'] += F.l1_loss(
                pred_matched[:, TrajParamIndex.X:TrajParamIndex.Z+1],
                gt_matched[:, TrajParamIndex.X:TrajParamIndex.Z+1],
                reduction='mean'
            )
            
            # Velocity loss (L1)
            losses['velocity'] += F.l1_loss(
                pred_matched[:, TrajParamIndex.VX:TrajParamIndex.VY+1],
                gt_matched[:, TrajParamIndex.VX:TrajParamIndex.VY+1],
                reduction='mean'
            )
            
            # Acceleration loss (L1)
            losses['acceleration'] += F.l1_loss(
                pred_matched[:, TrajParamIndex.AX:TrajParamIndex.AY+1],
                gt_matched[:, TrajParamIndex.AX:TrajParamIndex.AY+1],
                reduction='mean'
            )
            
            # Yaw loss (circular)
            yaw_diff = torch.abs(
                pred_matched[:, TrajParamIndex.YAW] - 
                gt_matched[:, TrajParamIndex.YAW]
            )
            yaw_diff = torch.min(yaw_diff, 2*torch.pi - yaw_diff)
            losses['yaw'] += yaw_diff.mean()
            
            # Size loss (L1)
            losses['size'] += F.l1_loss(
                pred_matched[:, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1],
                gt_matched[:, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1],
                reduction='mean'
            )
            
            # Object type loss (cross-entropy)
            # Convert object type to integer index
            pred_type = pred_matched[:, TrajParamIndex.OBJECT_TYPE]
            gt_type = gt_matched[:, TrajParamIndex.OBJECT_TYPE].long()
            losses['object_type'] += F.cross_entropy(
                pred_type.unsqueeze(1),
                gt_type
            )
            
            # Attributes loss (binary cross-entropy)
            attr_indices = [TrajParamIndex.HAS_OBJECT, TrajParamIndex.STATIC, TrajParamIndex.OCCLUDED]
            pred_attrs = pred_matched[:, attr_indices]
            gt_attrs = gt_matched[:, attr_indices]
            losses['attributes'] += F.binary_cross_entropy_with_logits(
                pred_attrs,
                gt_attrs,
                reduction='mean'
            )
        
        # Average over batches with valid matches
        valid_batch_count = 0
        for b in range(B):
            # Check if there are valid ground truth in this batch
            if gt_valid is not None:
                if gt_valid[b].sum() > 0:
                    valid_batch_count += 1
            else:
                if (gt_trajs[b, :, TrajParamIndex.HAS_OBJECT] > 0.5).sum() > 0:
                    valid_batch_count += 1
                    
        # Avoid division by zero
        valid_batch_count = max(1, valid_batch_count)
        
        for k in losses:
            losses[k] = losses[k] / valid_batch_count
            
        return losses

class PerceptionLoss(nn.Module):
    """Loss function for end-to-end 3D perception."""
    def __init__(self, weight_dict=None):
        super().__init__()
        self.weight_dict = weight_dict if weight_dict is not None else {
            'motion': 1.0,
            'type': 0.5,
            'attributes': 0.5,
            'existence': 2.0
        }
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: Dict from model with:
                - trajectories: List of ObstacleTrajectory objects
                - aux_trajectories: List of intermediate trajectory dicts containing:
                    - motion_params: Tensor[B, N, TrajParamIndex.END_OF_INDEX]
                    - object_type: Tensor[B, N] - class indices
                    - attributes: Tensor[B, N, AttributeType.END_OF_INDEX]
                    - type_logits: Tensor[B, N, len(ObjectType)]
            
            targets: Dict with:
                - gt_trajs: Tensor[B, M, TrajParamIndex.END_OF_INDEX] - Ground truth trajectories
        """
        # Get the final layer predictions
        pred_trajectories = outputs['aux_trajectories'][-1]
        
        # Get batch size
        B = pred_trajectories['motion_params'].shape[0]
        device = pred_trajectories['motion_params'].device
        
        # Initialize losses
        total_loss = torch.tensor(0.0, device=device)
        motion_loss = torch.tensor(0.0, device=device)
        type_loss = torch.tensor(0.0, device=device)
        attribute_loss = torch.tensor(0.0, device=device)
        existence_loss = torch.tensor(0.0, device=device)
        
        # Process each batch separately
        valid_batch_count = 0
        
        for b in range(B):
            # Extract predictions for this batch
            pred_motion = pred_trajectories['motion_params'][b]  # [N, TrajParamIndex.END_OF_INDEX]
            pred_type_logits = pred_trajectories['type_logits'][b]  # [N, num_classes]
            pred_attributes = pred_trajectories['attributes'][b]  # [N, AttributeType.END_OF_INDEX]
            
            # Extract ground truth for this batch
            gt_trajs = targets['gt_trajs'][b]  # [M, TrajParamIndex.END_OF_INDEX]
            
            # Filter valid ground truth trajectories (those with HAS_OBJECT > 0.5)
            gt_valid_mask = gt_trajs[:, TrajParamIndex.HAS_OBJECT] > 0.5
            gt_valid = gt_trajs[gt_valid_mask]  # [valid_M, TrajParamIndex.END_OF_INDEX]
            
            # Skip if no valid ground truth
            if len(gt_valid) == 0:
                continue
            
            valid_batch_count += 1
            
            # Get existence probability (from HAS_OBJECT attribute)
            pred_existence = pred_attributes[:, AttributeType.HAS_OBJECT]
            
            # Match predictions to ground truth using Hungarian algorithm
            # For simplicity, we just use the motion parameters for matching
            matcher = TrajectoryMatcher()
            indices, _ = matcher(pred_motion, gt_valid)
            
            # Skip if no matches
            if len(indices) == 0:
                # Still need to compute existence loss
                target_existence = torch.zeros_like(pred_existence)
                existence_loss += F.binary_cross_entropy_with_logits(
                    pred_existence, target_existence, reduction='mean'
                )
                continue
            
            # Reorder predictions to match ground truth
            matched_indices, gt_indices = zip(*indices)
            matched_motion = pred_motion[list(matched_indices)]
            matched_type_logits = pred_type_logits[list(matched_indices)]
            matched_attributes = pred_attributes[list(matched_indices)]
            
            # Get matched ground truth
            matched_gt = gt_valid[list(gt_indices)]
            
            # Compute motion loss (L2 distance for position, velocity, size)
            # Position (XYZ)
            pos_loss = F.mse_loss(
                matched_motion[:, TrajParamIndex.X:TrajParamIndex.Z+1],
                matched_gt[:, TrajParamIndex.X:TrajParamIndex.Z+1]
            )
            
            # Velocity (VX, VY)
            vel_loss = F.mse_loss(
                matched_motion[:, TrajParamIndex.VX:TrajParamIndex.VY+1],
                matched_gt[:, TrajParamIndex.VX:TrajParamIndex.VY+1]
            )
            
            # Size (LENGTH, WIDTH, HEIGHT)
            size_loss = F.mse_loss(
                matched_motion[:, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1],
                matched_gt[:, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1]
            )
            
            # Yaw (circular)
            yaw_diff = torch.abs(matched_motion[:, TrajParamIndex.YAW] - matched_gt[:, TrajParamIndex.YAW])
            yaw_loss = torch.min(yaw_diff, 2*torch.pi - yaw_diff).mean()
            
            # Combine motion losses
            batch_motion_loss = pos_loss + vel_loss + size_loss + yaw_loss
            motion_loss += batch_motion_loss
            
            # Compute object type loss
            # Get ground truth object type as integer indices
            gt_type = matched_gt[:, TrajParamIndex.OBJECT_TYPE].long()
            batch_type_loss = F.cross_entropy(matched_type_logits, gt_type)
            type_loss += batch_type_loss
            
            # Compute attribute loss (binary cross-entropy)
            # Get relevant attributes: STATIC, OCCLUDED
            attrs_idx = [TrajParamIndex.STATIC, TrajParamIndex.OCCLUDED]
            pred_attrs = matched_attributes[:, 1:3]  # Skip HAS_OBJECT (index 0)
            gt_attrs = matched_gt[:, attrs_idx]
            
            batch_attr_loss = F.binary_cross_entropy_with_logits(pred_attrs, gt_attrs)
            attribute_loss += batch_attr_loss
            
            # Compute existence loss
            # Create target: 1 for matched queries, 0 for unmatched
            target_existence = torch.zeros_like(pred_existence)
            target_existence[list(matched_indices)] = 1.0
            
            batch_existence_loss = F.binary_cross_entropy_with_logits(
                pred_existence, target_existence
            )
            existence_loss += batch_existence_loss
        
        # Average over batches with valid matches
        valid_batch_count = max(1, valid_batch_count)  # Avoid division by zero
        
        motion_loss /= valid_batch_count
        type_loss /= valid_batch_count
        attribute_loss /= valid_batch_count
        existence_loss /= valid_batch_count
        
        # Weighted sum of losses
        total_loss = (
            self.weight_dict['motion'] * motion_loss +
            self.weight_dict['type'] * type_loss +
            self.weight_dict['attributes'] * attribute_loss +
            self.weight_dict['existence'] * existence_loss
        )
        
        return {
            'loss': total_loss,
            'motion_loss': motion_loss,
            'type_loss': type_loss,
            'attribute_loss': attribute_loss,
            'existence_loss': existence_loss
        } 