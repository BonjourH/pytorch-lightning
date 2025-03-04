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

def match_trajectories(
    pred_trajs: torch.Tensor,
    gt_trajs: torch.Tensor,
    frames: int = 10,
    dt: float = 0.1,
    iou_method: str = "iou2"
) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
    """Match predicted trajectories to ground truth trajectories using Hungarian algorithm.
    
    Args:
        pred_trajs: Predicted trajectories tensor [N, params]
        gt_trajs: Ground truth trajectories tensor [M, params]
        frames: Number of frames to consider
        dt: Time delta between frames
        iou_method: Method to calculate IoU ("iou1" or "iou2")
        
    Returns:
        Tuple of:
        - List of matched indices (pred_idx, gt_idx)
        - Cost matrix used for matching
    """
    N, M = pred_trajs.shape[0], gt_trajs.shape[0]
    
    if N == 0 or M == 0:
        return [], torch.zeros((0, 0), device=pred_trajs.device)
    
    device = pred_trajs.device
    
    # Create frames tensor
    frames_tensor = torch.linspace(-dt * (frames - 1), 0, frames, device=device)
    
    # Initialize cost matrix
    cost_matrix = torch.zeros((N, M), device=device)
    
    # Calculate trajectory costs between all predictions and ground truth
    for i in range(N):
        for j in range(M):
            pred = pred_trajs[i]
            gt = gt_trajs[j]
            
            if iou_method == "iou1":
                iou = calculate_trajectory_bev_iou(pred, gt, frames_tensor)
            else:  # iou2 by default
                iou = calculate_trajectory_bev_iou2(pred, gt, frames_tensor)
                
            score = 0.8 * iou + 0.2 * calculate_trajectory_distance(pred, gt)
            
            cost_matrix[i, j] = 1000.0 * (1.0 - score)
    
    # Run Hungarian algorithm
    pred_idx, gt_idx = linear_sum_assignment(cost_matrix.cpu().numpy())
    indices = [(int(i), int(j)) for i, j in zip(pred_idx, gt_idx)]
    
    return indices, cost_matrix

class TrajectoryLoss(nn.Module):
    """Loss function for trajectory prediction."""
    
    def __init__(self, 
                 weight_dict=None,
                 frames: int = 10,
                 dt: float = 0.1,
                 iou_method: str = "iou2",
                 aux_loss_weight: float = 0.5):
        """Initialize the trajectory loss.
        
        Args:
            weight_dict: Dictionary of loss weights for different components
            frames: Number of frames to consider for trajectory matching
            dt: Time delta between frames
            iou_method: Method to calculate IoU between trajectories
            aux_loss_weight: Weight factor for auxiliary losses
        """
        super().__init__()
        self.weight_dict = weight_dict if weight_dict is not None else {
            'center': 1.0,
            'velocity': 1.0,
            'acceleration': 0.5,
            'yaw': 1.0,
            'size': 1.0,
            'object_type': 0.5,
            'attributes': 0.5,
            'existence': 2.0
        }
        self.frames = frames
        self.dt = dt
        self.iou_method = iou_method
        self.aux_loss_weight = aux_loss_weight
        
    def forward(self, outputs: List[Dict], targets: Dict) -> Dict:
        """Compute loss between predicted and ground truth trajectories.
        
        Args:
            outputs: List of dicts, each containing:
                - traj_params: Tensor[B, N, TrajParamIndex.END_OF_INDEX]
                - type_logits: Tensor[B, N, len(ObjectType)]
                The last element contains the final predictions.
            targets: Dict containing:
                - trajs: Tensor[B, M, TrajParamIndex.END_OF_INDEX] - ground truth
                
        Returns:
            Dict of losses
        """
        total_loss = 0.0
        losses = {}
        
        # Get ground truth trajectories
        gt_trajs = targets['trajs']  # [B, M, TrajParamIndex.END_OF_INDEX]
        
        # Decode object types from ground truth trajectories
        gt_types = gt_trajs[:, :, TrajParamIndex.OBJECT_TYPE].long()  # [B, M]
        
        # Process final predictions (last element in outputs list)
        final_pred = outputs[-1]
        pred_trajs = final_pred['traj_params']  # [B, N, TrajParamIndex.END_OF_INDEX]
        pred_types = final_pred['type_logits']  # [B, N, len(ObjectType)]
        
        # Compute main loss
        main_losses = self._compute_losses(pred_trajs, gt_trajs)
        
        # Compute object type loss
        type_loss = F.cross_entropy(pred_types.view(-1, pred_types.size(-1)), gt_types.view(-1), reduction='mean')
        main_losses['object_type'] = type_loss
        
        # Add to total loss with weights
        for k, v in main_losses.items():
            losses[k] = v
            total_loss += self.weight_dict.get(k, 1.0) * v
        
        # Process auxiliary predictions (all elements except the last one)
        if len(outputs) > 1:
            aux_losses = {}
            # Process each auxiliary output
            for i, aux_pred in enumerate(outputs[:-1]):
                layer_losses = self._compute_losses(
                    aux_pred['traj_params'], 
                    gt_trajs,
                    prefix=f'aux_{i}_'
                )
                
                # Compute auxiliary object type loss
                aux_type_loss = F.cross_entropy(
                    aux_pred['type_logits'].view(-1, aux_pred['type_logits'].size(-1)), 
                    gt_types.view(-1), 
                    reduction='mean'
                )
                layer_losses[f'aux_{i}_object_type'] = aux_type_loss
                
                # Add to auxiliary losses
                for k, v in layer_losses.items():
                    aux_losses[k] = v
                    # Apply a smaller weight to auxiliary losses
                    weight_factor = self.aux_loss_weight ** (len(outputs) - 2 - i)
                    total_loss += weight_factor * self.weight_dict.get(k.split('_')[-1], 1.0) * v
            
            # Add auxiliary losses to the main losses dict
            losses.update(aux_losses)
        
        # Add total loss
        losses['loss'] = total_loss
        
        return losses
    
    def _compute_losses(self, 
                       pred_trajs: torch.Tensor, 
                       gt_trajs: torch.Tensor,
                       prefix: str = '') -> Dict[str, torch.Tensor]:
        """Compute losses between predicted and ground truth trajectories.
        
        Args:
            pred_trajs: Tensor[B, N, TrajParamIndex.END_OF_INDEX] - Predictions
            gt_trajs: Tensor[B, M, TrajParamIndex.END_OF_INDEX] - Ground truth
            prefix: Optional prefix for loss keys
            
        Returns:
            Dict of losses
        """
        B = pred_trajs.shape[0]
        device = pred_trajs.device
        
        # Initialize losses
        losses = {
            f'{prefix}center': torch.tensor(0.0, device=device),
            f'{prefix}velocity': torch.tensor(0.0, device=device),
            f'{prefix}acceleration': torch.tensor(0.0, device=device),
            f'{prefix}yaw': torch.tensor(0.0, device=device),
            f'{prefix}size': torch.tensor(0.0, device=device),
            f'{prefix}object_type': torch.tensor(0.0, device=device),
            f'{prefix}attributes': torch.tensor(0.0, device=device),
            f'{prefix}existence': torch.tensor(0.0, device=device)
        }
        
        # Process each batch separately
        valid_batch_count = 0
        
        for b in range(B):
            # Get batch predictions
            b_pred = pred_trajs[b]  # [N, TrajParamIndex.END_OF_INDEX]
            
            # Get existence probability
            pred_existence = b_pred[:, TrajParamIndex.HAS_OBJECT]
            
            # Get batch ground truth
            b_gt = gt_trajs[b]  # [M, TrajParamIndex.END_OF_INDEX]
            
            # Filter valid ground truth using HAS_OBJECT flag
            gt_valid_mask = b_gt[:, TrajParamIndex.HAS_OBJECT] > 0.5
            b_gt_valid = b_gt[gt_valid_mask]  # [valid_M, TrajParamIndex.END_OF_INDEX]
            
            # Skip if no valid ground truth
            if len(b_gt_valid) == 0:
                continue
            
            valid_batch_count += 1
            
            # For existence loss, create a target tensor with 1s for valid GT count, 0s for rest
            target_existence = torch.zeros_like(pred_existence)
            target_existence[:len(b_gt_valid)] = 1.0
            losses[f'{prefix}existence'] += F.binary_cross_entropy_with_logits(
                pred_existence, target_existence, reduction='mean'
            )
            
            # Filter valid predictions (those with HAS_OBJECT > 0.5)
            pred_valid_mask = b_pred[:, TrajParamIndex.HAS_OBJECT] > 0.5
            b_pred_valid = b_pred[pred_valid_mask]  # [valid_N, TrajParamIndex.END_OF_INDEX]
            
            # Skip if no valid predictions
            if len(b_pred_valid) == 0:
                continue
            
            # Match predictions to ground truth
            indices, _ = match_trajectories(
                b_pred_valid, 
                b_gt_valid,
                self.frames,
                self.dt, 
                self.iou_method
            )
            
            if len(indices) == 0:
                continue
            
            # Extract matched pairs
            pred_indices, gt_indices = zip(*indices)
            pred_matched = b_pred_valid[list(pred_indices)]
            gt_matched = b_gt_valid[list(gt_indices)]
            
            # Center loss (L2)
            losses[f'{prefix}center'] += F.mse_loss(
                pred_matched[:, TrajParamIndex.X:TrajParamIndex.Z+1],
                gt_matched[:, TrajParamIndex.X:TrajParamIndex.Z+1],
                reduction='mean'
            )
            
            # Velocity loss (L2)
            losses[f'{prefix}velocity'] += F.mse_loss(
                pred_matched[:, TrajParamIndex.VX:TrajParamIndex.VY+1],
                gt_matched[:, TrajParamIndex.VX:TrajParamIndex.VY+1],
                reduction='mean'
            )
            
            # Acceleration loss (L1)
            losses[f'{prefix}acceleration'] += F.l1_loss(
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
            losses[f'{prefix}yaw'] += yaw_diff.mean()
            
            # Size loss (L1)
            losses[f'{prefix}size'] += F.l1_loss(
                pred_matched[:, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1],
                gt_matched[:, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1],
                reduction='mean'
            )
            
            # Object type loss (cross-entropy)
            pred_type = pred_matched[:, TrajParamIndex.OBJECT_TYPE]
            gt_type = gt_matched[:, TrajParamIndex.OBJECT_TYPE].long()
            losses[f'{prefix}object_type'] += F.cross_entropy(
                pred_type.unsqueeze(1),
                gt_type
            )
            
            # Attributes loss (binary cross-entropy)
            attr_indices = [TrajParamIndex.HAS_OBJECT, TrajParamIndex.STATIC, TrajParamIndex.OCCLUDED]
            pred_attrs = pred_matched[:, attr_indices]
            gt_attrs = gt_matched[:, attr_indices]
            losses[f'{prefix}attributes'] += F.binary_cross_entropy_with_logits(
                pred_attrs,
                gt_attrs,
                reduction='mean'
            )
        
        # Average over batches with valid trajectories
        valid_batch_count = max(1, valid_batch_count)  # Avoid division by zero
        
        for k in losses:
            losses[k] = losses[k] / valid_batch_count
            
        return losses

def calculate_trajectory_bev_iou(traj1: torch.Tensor, traj2: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between two trajectories in BEV space, considering their movement over time.
    
    Args:
        traj1: Tensor[TrajParamIndex.END_OF_INDEX] - First trajectory
        traj2: Tensor[TrajParamIndex.END_OF_INDEX] - Second trajectory
        frames: Tensor of timestamp values from past to present (e.g., [-0.9, -0.8, ..., 0])
        
    Returns:
        IoU value between 0 and 1
    """
    device = traj1.device
    
    # Extract parameters
    # Position at t=0
    x1, y1 = traj1[TrajParamIndex.X], traj1[TrajParamIndex.Y]
    x2, y2 = traj2[TrajParamIndex.X], traj2[TrajParamIndex.Y]
    
    # Velocity at t=0
    vx1, vy1 = traj1[TrajParamIndex.VX], traj1[TrajParamIndex.VY]
    vx2, vy2 = traj2[TrajParamIndex.VX], traj2[TrajParamIndex.VY]
    
    # Acceleration
    ax1, ay1 = traj1[TrajParamIndex.AX], traj1[TrajParamIndex.AY]
    ax2, ay2 = traj2[TrajParamIndex.AX], traj2[TrajParamIndex.AY]
    
    # Object dimensions
    l1, w1 = traj1[TrajParamIndex.LENGTH], traj1[TrajParamIndex.WIDTH]
    l2, w2 = traj2[TrajParamIndex.LENGTH], traj2[TrajParamIndex.WIDTH]
    
    # Yaw at t=0
    yaw1 = traj1[TrajParamIndex.YAW]
    yaw2 = traj2[TrajParamIndex.YAW]
    
    # Initialize arrays to store bounding boxes for each time step
    boxes1 = []
    boxes2 = []
    
    # Calculate bounding boxes at each time step
    for t in frames:
        # Calculate position at time t using motion equations
        # x(t) = x0 + v0*t + 0.5*a*t^2
        pos_x1 = x1 + vx1 * t + 0.5 * ax1 * t * t
        pos_y1 = y1 + vy1 * t + 0.5 * ay1 * t * t
        
        pos_x2 = x2 + vx2 * t + 0.5 * ax2 * t * t
        pos_y2 = y2 + vy2 * t + 0.5 * ay2 * t * t
        
        # Calculate velocity at time t
        # v(t) = v0 + a*t
        vel_x1 = vx1 + ax1 * t
        vel_y1 = vy1 + ay1 * t
        
        vel_x2 = vx2 + ax2 * t
        vel_y2 = vy2 + ay2 * t
        
        # Determine yaw based on velocity or use initial yaw
        speed1 = torch.sqrt(vel_x1*vel_x1 + vel_y1*vel_y1)
        speed2 = torch.sqrt(vel_x2*vel_x2 + vel_y2*vel_y2)
        
        # Use velocity direction if speed is sufficient, otherwise use provided yaw
        yaw_t1 = torch.where(speed1 > 0.2, torch.atan2(vel_y1, vel_x1), yaw1)
        yaw_t2 = torch.where(speed2 > 0.2, torch.atan2(vel_y2, vel_x2), yaw2)
        
        # Calculate corners of bounding boxes
        # For each box, we calculate 4 corners in BEV
        cos_yaw1, sin_yaw1 = torch.cos(yaw_t1), torch.sin(yaw_t1)
        cos_yaw2, sin_yaw2 = torch.cos(yaw_t2), torch.sin(yaw_t2)
        
        # Half dimensions
        hl1, hw1 = l1/2, w1/2
        hl2, hw2 = l2/2, w2/2
        
        # Corners for box 1 (front-left, front-right, rear-right, rear-left)
        corners1 = torch.tensor([
            [pos_x1 + hl1*cos_yaw1 - hw1*sin_yaw1, pos_y1 + hl1*sin_yaw1 + hw1*cos_yaw1],
            [pos_x1 + hl1*cos_yaw1 + hw1*sin_yaw1, pos_y1 + hl1*sin_yaw1 - hw1*cos_yaw1],
            [pos_x1 - hl1*cos_yaw1 + hw1*sin_yaw1, pos_y1 - hl1*sin_yaw1 - hw1*cos_yaw1],
            [pos_x1 - hl1*cos_yaw1 - hw1*sin_yaw1, pos_y1 - hl1*sin_yaw1 + hw1*cos_yaw1]
        ], device=device)
        
        # Corners for box 2
        corners2 = torch.tensor([
            [pos_x2 + hl2*cos_yaw2 - hw2*sin_yaw2, pos_y2 + hl2*sin_yaw2 + hw2*cos_yaw2],
            [pos_x2 + hl2*cos_yaw2 + hw2*sin_yaw2, pos_y2 + hl2*sin_yaw2 - hw2*cos_yaw2],
            [pos_x2 - hl2*cos_yaw2 + hw2*sin_yaw2, pos_y2 - hl2*sin_yaw2 - hw2*cos_yaw2],
            [pos_x2 - hl2*cos_yaw2 - hw2*sin_yaw2, pos_y2 - hl2*sin_yaw2 + hw2*cos_yaw2]
        ], device=device)
        
        boxes1.append(corners1)
        boxes2.append(corners2)
    
    # For simplification, we approximate the shadow as the convex hull
    # of all bounding boxes. For a rough approximation, we can use the
    # min/max extents of all corners as a bounding box of the shadow.
    
    # Stack all corners from all time steps
    all_corners1 = torch.cat(boxes1, dim=0)  # [(time_steps+1)*4, 2]
    all_corners2 = torch.cat(boxes2, dim=0)  # [(time_steps+1)*4, 2]
    
    # Get min/max extents
    min_x1, min_y1 = torch.min(all_corners1[:, 0]), torch.min(all_corners1[:, 1])
    max_x1, max_y1 = torch.max(all_corners1[:, 0]), torch.max(all_corners1[:, 1])
    
    min_x2, min_y2 = torch.min(all_corners2[:, 0]), torch.min(all_corners2[:, 1])
    max_x2, max_y2 = torch.max(all_corners2[:, 0]), torch.max(all_corners2[:, 1])
    
    # Calculate intersection
    inter_x1 = torch.maximum(min_x1, min_x2)
    inter_y1 = torch.maximum(min_y1, min_y2)
    inter_x2 = torch.minimum(max_x1, max_x2)
    inter_y2 = torch.minimum(max_y1, max_y2)
    
    # Clamp to ensure valid intersection
    inter_width = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_height = torch.clamp(inter_y2 - inter_y1, min=0.0)
    
    # Calculate areas
    inter_area = inter_width * inter_height
    area1 = (max_x1 - min_x1) * (max_y1 - min_y1)
    area2 = (max_x2 - min_x2) * (max_y2 - min_y2)
    union_area = area1 + area2 - inter_area
    
    # Avoid division by zero
    union_area = torch.clamp(union_area, min=1e-8)
    
    # Return IoU
    iou = inter_area / union_area
    return iou 

def calculate_trajectory_bev_iou2(traj1: torch.Tensor, traj2: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between two trajectories in BEV space, simplified as linear motion considering only start and end points.
    
    Trajectory is represented as a large rotated rectangle:
    - Length is increased by the distance of x-direction movement
    - Width remains the original width
    - Center point is at the midpoint of start and end positions
    - Rotation angle is determined by velocity direction at the midpoint
    - If velocity is small or the object is static, the original yaw is used
    
    Args:
        traj1: Tensor[TrajParamIndex.END_OF_INDEX] - First trajectory
        traj2: Tensor[TrajParamIndex.END_OF_INDEX] - Second trajectory
        frames: Tensor - Timestamp values from past to present (e.g., [-0.9, -0.8, ..., 0])
        
    Returns:
        IoU value between 0 and 1
    """
    device = traj1.device
    
    # Get start and end times
    start_time = frames[0]  # Earliest time (-t)
    end_time = frames[-1]   # Latest time (0)
    
    # Extract trajectory 1 parameters
    # Current position (t=0)
    x1_end, y1_end = traj1[TrajParamIndex.X], traj1[TrajParamIndex.Y]
    
    # Velocity and acceleration
    vx1, vy1 = traj1[TrajParamIndex.VX], traj1[TrajParamIndex.VY]
    ax1, ay1 = traj1[TrajParamIndex.AX], traj1[TrajParamIndex.AY]
    
    # Calculate start position (t=-T)
    # x(t) = x0 + v0*t + 0.5*a*t^2
    x1_start = x1_end + vx1 * start_time + 0.5 * ax1 * start_time * start_time
    y1_start = y1_end + vy1 * start_time + 0.5 * ay1 * start_time * start_time
    
    # Original dimensions of trajectory 1
    l1, w1 = traj1[TrajParamIndex.LENGTH], traj1[TrajParamIndex.WIDTH]
    
    # Original direction of trajectory 1
    yaw1_orig = traj1[TrajParamIndex.YAW]
    
    # Calculate midpoint position
    x1_mid = (x1_start + x1_end) / 2
    y1_mid = (y1_start + y1_end) / 2
    
    # Calculate midpoint velocity
    mid_time = (start_time + end_time) / 2
    vx1_mid = vx1 + ax1 * mid_time
    vy1_mid = vy1 + ay1 * mid_time
    
    # Calculate movement distance
    dx1 = x1_end - x1_start
    dy1 = y1_end - y1_start
    
    # Calculate movement length
    move_dist1 = torch.sqrt(dx1*dx1 + dy1*dy1)
    
    # Calculate extended dimensions
    l1_ext = l1 + move_dist1
    w1_ext = w1
    
    # Determine direction
    speed1_mid = torch.sqrt(vx1_mid*vx1_mid + vy1_mid*vy1_mid)
    static1 = traj1[TrajParamIndex.STATIC] > 0.5
    
    # If speed is sufficient and not static, use velocity direction; otherwise use original yaw
    yaw1 = torch.where((speed1_mid > 0.2) & (~static1), 
                       torch.atan2(vy1_mid, vx1_mid), 
                       yaw1_orig)
    
    # Extract trajectory 2 parameters
    # Current position (t=0)
    x2_end, y2_end = traj2[TrajParamIndex.X], traj2[TrajParamIndex.Y]
    
    # Velocity and acceleration
    vx2, vy2 = traj2[TrajParamIndex.VX], traj2[TrajParamIndex.VY]
    ax2, ay2 = traj2[TrajParamIndex.AX], traj2[TrajParamIndex.AY]
    
    # Calculate start position (t=-T)
    x2_start = x2_end + vx2 * start_time + 0.5 * ax2 * start_time * start_time
    y2_start = y2_end + vy2 * start_time + 0.5 * ay2 * start_time * start_time
    
    # Original dimensions of trajectory 2
    l2, w2 = traj2[TrajParamIndex.LENGTH], traj2[TrajParamIndex.WIDTH]
    
    # Original direction of trajectory 2
    yaw2_orig = traj2[TrajParamIndex.YAW]
    
    # Calculate midpoint position
    x2_mid = (x2_start + x2_end) / 2
    y2_mid = (y2_start + y2_end) / 2
    
    # Calculate midpoint velocity
    vx2_mid = vx2 + ax2 * mid_time
    vy2_mid = vy2 + ay2 * mid_time
    
    # Calculate movement distance
    dx2 = x2_end - x2_start
    dy2 = y2_end - y2_start
    
    # Calculate movement length
    move_dist2 = torch.sqrt(dx2*dx2 + dy2*dy2)
    
    # Calculate extended dimensions
    l2_ext = l2 + move_dist2
    w2_ext = w2
    
    # Determine direction
    speed2_mid = torch.sqrt(vx2_mid*vx2_mid + vy2_mid*vy2_mid)
    static2 = traj2[TrajParamIndex.STATIC] > 0.5
    
    # If speed is sufficient and not static, use velocity direction; otherwise use original yaw
    yaw2 = torch.where((speed2_mid > 0.2) & (~static2), 
                       torch.atan2(vy2_mid, vx2_mid), 
                       yaw2_orig)
    
    # Calculate IoU between two rotated rectangles
    # For simplification, we use an approximate method:
    # 1. Calculate the four corner points of each rotated rectangle
    # 2. Use axis-aligned bounding box approximation
    
    # Calculate corners of rotated rectangle 1
    cos_yaw1, sin_yaw1 = torch.cos(yaw1), torch.sin(yaw1)
    
    # Half of extended dimensions
    hl1, hw1 = l1_ext/2, w1_ext/2
    
    # Four corners of rectangle 1 (front-left, front-right, rear-right, rear-left)
    corners1 = torch.tensor([
        [x1_mid + hl1*cos_yaw1 - hw1*sin_yaw1, y1_mid + hl1*sin_yaw1 + hw1*cos_yaw1],
        [x1_mid + hl1*cos_yaw1 + hw1*sin_yaw1, y1_mid + hl1*sin_yaw1 - hw1*cos_yaw1],
        [x1_mid - hl1*cos_yaw1 + hw1*sin_yaw1, y1_mid - hl1*sin_yaw1 - hw1*cos_yaw1],
        [x1_mid - hl1*cos_yaw1 - hw1*sin_yaw1, y1_mid - hl1*sin_yaw1 + hw1*cos_yaw1]
    ], device=device)
    
    # Calculate corners of rotated rectangle 2
    cos_yaw2, sin_yaw2 = torch.cos(yaw2), torch.sin(yaw2)
    
    # Half of extended dimensions
    hl2, hw2 = l2_ext/2, w2_ext/2
    
    # Four corners of rectangle 2
    corners2 = torch.tensor([
        [x2_mid + hl2*cos_yaw2 - hw2*sin_yaw2, y2_mid + hl2*sin_yaw2 + hw2*cos_yaw2],
        [x2_mid + hl2*cos_yaw2 + hw2*sin_yaw2, y2_mid + hl2*sin_yaw2 - hw2*cos_yaw2],
        [x2_mid - hl2*cos_yaw2 + hw2*sin_yaw2, y2_mid - hl2*sin_yaw2 - hw2*cos_yaw2],
        [x2_mid - hl2*cos_yaw2 - hw2*sin_yaw2, y2_mid - hl2*sin_yaw2 + hw2*cos_yaw2]
    ], device=device)
    
    # For rotated rectangles IoU calculation, we use bounding box approximation:
    # Find min/max coordinates of all corners to create bounding boxes
    
    # Bounding box for trajectory 1
    min_x1, min_y1 = torch.min(corners1[:, 0]), torch.min(corners1[:, 1])
    max_x1, max_y1 = torch.max(corners1[:, 0]), torch.max(corners1[:, 1])
    
    # Bounding box for trajectory 2
    min_x2, min_y2 = torch.min(corners2[:, 0]), torch.min(corners2[:, 1])
    max_x2, max_y2 = torch.max(corners2[:, 0]), torch.max(corners2[:, 1])
    
    # Calculate intersection
    inter_x1 = torch.maximum(min_x1, min_x2)
    inter_y1 = torch.maximum(min_y1, min_y2)
    inter_x2 = torch.minimum(max_x1, max_x2)
    inter_y2 = torch.minimum(max_y1, max_y2)
    
    # Ensure valid intersection
    inter_width = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_height = torch.clamp(inter_y2 - inter_y1, min=0.0)
    
    # Calculate areas
    inter_area = inter_width * inter_height
    area1 = (max_x1 - min_x1) * (max_y1 - min_y1)
    area2 = (max_x2 - min_x2) * (max_y2 - min_y2)
    union_area = area1 + area2 - inter_area
    
    # Avoid division by zero
    union_area = torch.clamp(union_area, min=1e-8)
    
    # Return IoU
    iou = inter_area / union_area
    return iou 

def calculate_trajectory_distance(traj1, traj2):
    """
    Calculate the distance between two trajectories based on their positions at different timestamps.
    
    Args:
        traj1: Tensor[TrajParamIndex.END_OF_INDEX] - First trajectory
        traj2: Tensor[TrajParamIndex.END_OF_INDEX] - Second trajectory
        
    Returns:
        float: Distance between the two trajectories
    """
    # Calculate distance score using L2 distance normalized by object size
    pos_diff = torch.sqrt((traj1[TrajParamIndex.X: TrajParamIndex.Z + 1] - traj2[TrajParamIndex.X: TrajParamIndex.Z + 1])**2)
    # Use average size of both objects to normalize distance
    avg_size = (torch.sqrt(traj1[TrajParamIndex.LENGTH]**2 + traj1[TrajParamIndex.WIDTH]**2) + 
              torch.sqrt(traj2[TrajParamIndex.LENGTH]**2 + traj2[TrajParamIndex.WIDTH]**2)) / 2
    # Normalized distance score that decays with distance relative to object size
    dist_score = torch.exp(-pos_diff / (avg_size + 1e-6))
    return dist_score