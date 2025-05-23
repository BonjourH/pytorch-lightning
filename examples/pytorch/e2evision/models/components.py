import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional, List
import numpy as np
import torch.nn.functional as F
from timm.models import create_model

from base import SourceCameraId, TrajParamIndex, CameraParamIndex, EgoStateIndex, CameraType
from utils.math_utils import generate_unit_cube_points, generate_bbox_corners_points, sample_bbox_edge_points

class TrajectoryQueryRefineLayer(nn.Module):
    """Single layer of trajectory decoder."""
    def __init__(self, feature_dim: int, num_heads: int = 8, dim_feedforward: int = 2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        
        self.linear1 = nn.Linear(feature_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, feature_dim)
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()
        
    def forward(self, 
                queries: torch.Tensor,
                memory: torch.Tensor,
                query_pos: Optional[torch.Tensor] = None,
                memory_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            queries: Object queries [B, num_queries, C]
            memory: Image features [B, H*W, C]
            query_pos: Query position encoding
            memory_pos: Memory position encoding
        Returns:
            Updated queries [B, num_queries, C]
        """
        # Self attention
        q = queries + query_pos if query_pos is not None else queries
        k = q
        v = queries
        queries2 = self.self_attn(q, k, v)[0]
        queries = self.norm1(queries + self.dropout(queries2))
        
        # Cross attention
        q = queries + query_pos if query_pos is not None else queries
        k = memory + memory_pos if memory_pos is not None else memory
        v = memory
        queries2 = self.cross_attn(q, k, v)[0]
        queries = self.norm2(queries + self.dropout(queries2))
        
        # Feed forward
        queries2 = self.linear2(self.dropout(self.activation(self.linear1(queries))))
        queries = self.norm3(queries + self.dropout(queries2))
        
        return queries 

    
class TrajectoryDecoder(nn.Module):
    """Decode trajectories from features."""
    
    def __init__(self,
                 num_layers: int = 6,
                 num_queries: int = 128,
                 feature_dim: int = 256,
                 hidden_dim: int = 512,
                 num_points: int = 25): # Points to sample per face of the unit cube
        super().__init__()
        
        query_dim = feature_dim
        
        # Object queries
        self.queries = nn.Parameter(torch.randn(num_queries, query_dim))
        
        # Query position encoding
        self.query_pos = nn.Parameter(torch.randn(num_queries, query_dim))
        
        # Sample points on unit cube for feature gathering
        # self.register_buffer('unit_points', generate_unit_cube_points(num_points)) # [P, 3]
        self.register_buffer('unit_points', generate_bbox_corners_points()) # [3, 9] corners + center
        self.register_buffer('origin_point', torch.zeros(3, 1)) # [3, 1]
        
        # Parameter ranges for normalization: torch.Tensor[TrajParamIndex.HEIGHT + 1, 2]
        ranges = self._get_motion_param_range()
        self.register_buffer('motion_min_vals', ranges[:, 0])
        self.register_buffer('motion_ranges', ranges[:, 1] - ranges[:, 0])
         
        # Single trajectory parameter head that outputs all trajectory parameters        
        self.motion_head = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, TrajParamIndex.HEIGHT + 1),
            nn.Sigmoid()
        )
        self.cls_head = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, TrajParamIndex.END_OF_INDEX - TrajParamIndex.HEIGHT - 1)
        )
        
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        
        # Refiner layers
        self.layers = nn.ModuleList([
            TrajectoryQueryRefineLayer(
                feature_dim=feature_dim,
                num_heads=8,
                dim_feedforward=hidden_dim
            ) for _ in range(num_layers)
        ])
        
        
    
    def decode_trajectory(self, x: torch.Tensor) -> torch.Tensor:
        """ Decode trajectory from features."""
        # 原始前向计算
        motion_params = self.motion_head(x)
        motion_params = motion_params * self.motion_ranges + self.motion_min_vals
        cls_params = self.cls_head(x)
        
        traj_params = torch.cat([motion_params, cls_params], dim=-1)
        return traj_params
    
    def forward(self, 
                features_dict: Dict[SourceCameraId, torch.Tensor],
                calibrations: Dict[SourceCameraId, torch.Tensor],
                ego_states: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            features_dict: Dict[camera_id -> Tensor[B, T, C, H, W]]
            calibrations: Dict[camera_id -> Tensor[B, CameraParamIndex.END_OF_INDEX]]
            ego_states: Tensor[B, T, EgoStateIndex.END_OF_INDEX]
            
        Returns:
            List of trajectory parameter tensors [B, num_queries, TrajParamIndex.END_OF_INDEX]
        """
        B = next(iter(features_dict.values())).shape[0]
        # Create initial object queries
        queries = self.queries.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, C]
        pos = self.query_pos.unsqueeze(0).repeat(B, 1, 1)    # [B, num_queries, C]
        
        # List to store all trajectory parameters
        outputs = []
        
        # Decoder iteratively refines trajectory parameters
        for layer in self.layers:
            # 1. Predict parameters from current queries
            traj_params = self.decode_trajectory(queries) 
            # traj_params: Tensor[B, N, TrajParamIndex.END_OF_INDEX]
            outputs.append(traj_params)
            
            # 2. Sample points on predicted objects
            box_points = self.sample_box_points(traj_params)  # [B, N, num_points, 3]
            
            # 3. Gather features from all views and frames
            point_features, validity_mask = self.gather_point_features(
                box_points, traj_params, features_dict, calibrations, ego_states
            )  # [B, N, num_points, T, num_cameras, C], [B, N, num_points, T, num_cameras, 1]
            
            # 4. Aggregate features
            agg_features = self.aggregate_features((point_features, validity_mask))  # [B, N, hidden_dim]
            
            # 5. Update queries
            queries = layer(queries, agg_features)
            
            
        # Get final predictions
        outputs.append(self.decode_trajectory(queries))
        
        return outputs
    
    def _get_motion_param_range(self)->torch.Tensor:
        """Get parameter ranges for normalization.
        
        Returns:
            Tensor of shape [TrajParamIndex.HEIGHT + 1, 2] containing min/max values
        """
        param_range = torch.zeros(TrajParamIndex.HEIGHT + 1, 2)
        
        # Position ranges (in meters)
        param_range[TrajParamIndex.X] = torch.tensor([-80.0, 250.0])
        param_range[TrajParamIndex.Y] = torch.tensor([-10.0, 10.0])
        param_range[TrajParamIndex.Z] = torch.tensor([-3.0, 5.0])
        
         # Velocity ranges (in m/s)
        param_range[TrajParamIndex.VX] = torch.tensor([-40.0, 40.0])
        param_range[TrajParamIndex.VY] = torch.tensor([-5.0, 5.0])
        
        # Acceleration ranges (in m/s^2)
        param_range[TrajParamIndex.AX] = torch.tensor([-5.0, 5.0])
        param_range[TrajParamIndex.AY] = torch.tensor([-2.0, 2.0])
        
        # Yaw range (in radians)
        param_range[TrajParamIndex.YAW] = torch.tensor([-np.pi, np.pi])
        
        # Dimension ranges (in meters)
        param_range[TrajParamIndex.LENGTH] = torch.tensor([0.2, 25.0])
        param_range[TrajParamIndex.WIDTH] = torch.tensor([0.2, 3.0])
        param_range[TrajParamIndex.HEIGHT] = torch.tensor([0.5, 5.0])
        
        return param_range
    
    def gather_point_features(self, 
                            box_points: torch.Tensor, 
                            traj_params: torch.Tensor,
                            features_dict: Dict[SourceCameraId, torch.Tensor],
                            calibrations: Dict[SourceCameraId, torch.Tensor],
                            ego_states: torch.Tensor) -> tuple:
        """Gather features for box points from all cameras and frames.
        
        Args:
            box_points: Tensor[B, N, num_points, 3]: Points in object local coordinates
            traj_params: Tensor[B, N, TrajParamIndex.END_OF_INDEX]
            features_dict: Dict[camera_id -> Tensor[B, C, H, W]]
            calibrations: Dict[camera_id -> Tensor[B, CameraParamIndex.END_OF_INDEX]]
            ego_states: Tensor[B, T, EgoStateIndex.END_OF_INDEX]
            
        Returns:
            Tuple of:
                Tensor[B, N, num_points, T, num_cameras, C] - Features
                Tensor[B, N, num_points, T, num_cameras, 1] - Validity mask (1=valid, 0=invalid)
        """
        B, N, P = box_points.shape[:3]
        T = ego_states.shape[1]
        num_cameras = len(features_dict)
        
        # Initialize output tensor to store all features
        C = next(iter(features_dict.values())).shape[1]
        all_point_features = torch.zeros(B, N, P, T, num_cameras, C).to(box_points.device)
        # 创建特征有效性掩码，1表示有效，0表示无效
        validity_mask = torch.zeros(B, N, P, T, num_cameras, 1).to(box_points.device)
        
        # For each time step, transform points to global frame, then project to cameras
        for t in range(T):
            # 1. Calculate object positions at time t
            # 确保时间戳计算使用高精度，避免小数点后的精度丢失
            dt = (ego_states[:, t, EgoStateIndex.TIMESTAMP] - ego_states[:, -1, EgoStateIndex.TIMESTAMP]).to(torch.float64)  # Time diff from reference frame
            dt = dt.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            
            # Calculate positions of points at time t
            # Use trajectory motion model to compute positions
            world_points = self.transform_points_at_time(box_points, traj_params, ego_states[:, t], dt)
            
            # Project points to each camera
            for cam_idx, (camera_id, features) in enumerate(features_dict.items()):
                # Get calibration parameters
                calib = calibrations[camera_id]  # [B, CameraParamIndex.END_OF_INDEX]
                
                # Project points to image
                points_2d = self.project_points_to_image(world_points, calib)  # [B, N, P, 2]
                
                # Check visibility - points outside [0,1] are considered invisible
                visible = (
                    (points_2d[..., 0] >= 0) & 
                    (points_2d[..., 0] < 1) & 
                    (points_2d[..., 1] >= 0) & 
                    (points_2d[..., 1] < 1)
                )  # [B, N, P]
                
                # Sample features at projected points
                H, W = features.shape[-2:]
                
                # Convert normalized coordinates [0,1] to grid coordinates [-1,1] for grid_sample
                norm_points = torch.zeros_like(points_2d)
                norm_points[..., 0] = 2.0 * points_2d[..., 0] - 1.0
                norm_points[..., 1] = 2.0 * points_2d[..., 1] - 1.0
                
                # Reshape for grid_sample
                grid = norm_points.view(B, N * P, 1, 2)
                grid = grid.to(dtype=features.dtype) # float64 → float32
                # Sample features
                # features: [B, C, H, W], grid: [B, N*P, 1, 2]
                sampled = F.grid_sample(
                    features, grid, mode='bilinear', 
                    padding_mode='zeros', align_corners=True
                )  # [B, C, N*P, 1]
                
                # Reshape back
                point_features = sampled.permute(0, 2, 3, 1).view(B, N, P, C)
                
                # 记录有效特征
                validity_mask[:, :, :, t, cam_idx, 0] = visible.float()
                
                # 将无效特征设为0，保留原始实现
                point_features = point_features * visible.unsqueeze(-1).float()
                
                # Store in output tensor
                all_point_features[:, :, :, t, cam_idx] = point_features
        
        return all_point_features, validity_mask
        
    def transform_points_at_time(self, 
                               box_points: torch.Tensor, 
                               traj_params: torch.Tensor,
                               ego_state: torch.Tensor, 
                               dt: float) -> torch.Tensor:
        """Transform box points to world frame at a specific time.
        
        Args:
            box_points: Tensor[B, N, P, 3] - Points in object local coordinates
            traj_params: Tensor[B, N, TrajParamIndex.END_OF_INDEX] - Trajectory parameters
            ego_state: Tensor[B, EgoStateIndex.END_OF_INDEX] - Ego vehicle state
            dt: float - Time difference from current frame
            
        Returns:
            Tensor[B, N, P, 3] - Points in world coordinates at time t
        """
        B, N, P = box_points.shape[:3]
        
        # Extract trajectory parameters
        pos_x = traj_params[..., TrajParamIndex.X].unsqueeze(-1)  # [B, N, 1]
        pos_y = traj_params[..., TrajParamIndex.Y].unsqueeze(-1)  # [B, N, 1]
        pos_z = traj_params[..., TrajParamIndex.Z].unsqueeze(-1)  # [B, N, 1]
        
        vel_x = traj_params[..., TrajParamIndex.VX].unsqueeze(-1)  # [B, N, 1]
        vel_y = traj_params[..., TrajParamIndex.VY].unsqueeze(-1)  # [B, N, 1]
        
        acc_x = traj_params[..., TrajParamIndex.AX].unsqueeze(-1)  # [B, N, 1]
        acc_y = traj_params[..., TrajParamIndex.AY].unsqueeze(-1)  # [B, N, 1]
        
        yaw = traj_params[..., TrajParamIndex.YAW].unsqueeze(-1)  # [B, N, 1]
        
        # Calculate position at time t using motion model (constant acceleration)
        # x(t) = x0 + v0*t + 0.5*a*t^2
        pos_x_t = pos_x + vel_x * dt + 0.5 * acc_x * dt * dt
        pos_y_t = pos_y + vel_y * dt + 0.5 * acc_y * dt * dt
        pos_z_t = pos_z  # Assume constant height
        
        # Calculate velocity at time t
        # v(t) = v0 + a*t
        vel_x_t = vel_x + acc_x * dt
        vel_y_t = vel_y + acc_y * dt
        
        # Determine yaw based on velocity or use initial yaw
        speed_t = torch.sqrt(vel_x_t*vel_x_t + vel_y_t*vel_y_t)
        
        # If speed is sufficient, use velocity direction; otherwise use provided yaw
        yaw_t = torch.where(speed_t > 0.2, torch.atan2(vel_y_t, vel_x_t), yaw)
        
        # Calculate rotation matrices
        cos_yaw = torch.cos(yaw_t)
        sin_yaw = torch.sin(yaw_t)
        
        # Rotate points
        local_x = box_points[..., 0]
        local_y = box_points[..., 1]
        local_z = box_points[..., 2]
        
        # Apply rotation
        global_x = local_x * cos_yaw - local_y * sin_yaw + pos_x_t
        global_y = local_x * sin_yaw + local_y * cos_yaw + pos_y_t
        global_z = local_z + pos_z_t
        
        # Combine coordinates
        global_points = torch.stack([global_x, global_y, global_z], dim=-1)
        
        # Transform to ego frame
        ego_points = self.transform_points_to_ego_frame(global_points, ego_state)
        
        return ego_points
    
    def transform_points_to_ego_frame(self, 
                                    points: torch.Tensor,
                                    ego_state: torch.Tensor) -> torch.Tensor:
        """Transform 3D points to ego vehicle frame at given time.
        
        Args:
            points: Tensor[B, N, P, 3]
            ego_state: Tensor[B, EgoStateIndex.END_OF_INDEX] with position, yaw, timestamp
            
        Returns:
            Tensor[B, N, P, 3]
        """
        B, N, P = points.shape[:3]
        
        # Extract ego position and yaw
        ego_yaw = ego_state[..., EgoStateIndex.YAW].unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
        
        # Create rotation matrix
        cos_yaw = torch.cos(ego_yaw)
        sin_yaw = torch.sin(ego_yaw)
        
        x_rotated = points[..., 0] * cos_yaw - points[..., 1] * sin_yaw # [B, N, 1]
        y_rotated = points[..., 0] * sin_yaw + points[..., 1] * cos_yaw # [B, N, 1]
        
        ego_x = ego_state[..., EgoStateIndex.X].unsqueeze(1).unsqueeze(2) # [B, 1, 1]
        ego_y = ego_state[..., EgoStateIndex.Y].unsqueeze(1).unsqueeze(2) # [B, 1, 1]
        
        x_rotated += ego_x
        y_rotated += ego_y
        
        results = torch.stack(
            [x_rotated, y_rotated, points[..., 2]], 
            dim=-1
        )
        
        return results
    def aggregate_features(self, point_features_with_mask: tuple) -> torch.Tensor:
        """Aggregate features from all points, frames and cameras.
        
        Args:
            point_features_with_mask: Tuple containing:
                - point_features: Tensor[B, N, P, T, num_cameras, C]
                - validity_mask: Tensor[B, N, P, T, num_cameras, 1]
            
        Returns:
            Tensor[B, N, hidden_dim]
        """
        point_features, validity_mask = point_features_with_mask
        B, N = point_features.shape[:2]
        C = point_features.shape[-1]  # 获取特征通道数
        
        # Reshape for processing
        features_flat = point_features.flatten(2, 4)  # [B, N, P*T*num_cameras, C]
        mask_flat = validity_mask.flatten(2, 4)  # [B, N, P*T*num_cameras, 1]
        
        # 使用掩码处理特征聚合
        # 将无效特征替换为非常小的负值，确保在max操作中不会被选中
        # masked_value应该小于模型可能产生的任何有效特征值
        masked_value = -1e9  # 一个足够小的值
        masked_features = torch.where(
            mask_flat > 0.5,
            features_flat,
            torch.full_like(features_flat, masked_value)
        )
        
        # Max pooling over all points, frames and cameras
        pooled_features, _ = torch.max(masked_features, dim=2)  # [B, N, C]
        
        # 检查每个通道是否至少有一个有效值
        # 使用正确的维度进行广播
        # 首先沿着特征维度对掩码求和，获得每个位置是否有任何有效的点
        has_valid_point = (mask_flat.sum(dim=2) > 0)  # [B, N, 1]
        
        # 使用 torch.where 代替乘法和加法操作，避免广播问题
        pooled_features = torch.where(
            has_valid_point,  # [B, N, 1]
            pooled_features,  # [B, N, C]
            torch.zeros_like(pooled_features)  # [B, N, C]
        )
        
        # Process through MLP
        processed_features = self.feature_mlp(pooled_features)  # [B, N, hidden_dim]
        
        return processed_features
    
    def project_points_to_image(self, points_3d: torch.Tensor, calib_params: torch.Tensor) -> torch.Tensor:
        """Project 3D points to image coordinates.
        
        Args:
            points_3d: Tensor[B, N, P, 3] - Points in ego coordinates
            calib_params: Tensor[B, CameraParamIndex.END_OF_INDEX] - Camera parameters
            
        Returns:
            Tensor[B, N, P, 2] - Points in normalized image coordinates [0,1]
        """
        B, N, P, _ = points_3d.shape
        
        # Reshape for batch processing
        points_flat = points_3d.view(B, N * P, 3)
        
        # Extract camera parameters
        camera_type = calib_params[:, CameraParamIndex.CAMERA_TYPE].long()
        
        # Get intrinsic parameters
        fx = calib_params[:, CameraParamIndex.FX].unsqueeze(1)  # [B, 1]
        fy = calib_params[:, CameraParamIndex.FY].unsqueeze(1)  # [B, 1]
        cx = calib_params[:, CameraParamIndex.CX].unsqueeze(1)  # [B, 1]
        cy = calib_params[:, CameraParamIndex.CY].unsqueeze(1)  # [B, 1]
        
        # Distortion parameters
        k1 = calib_params[:, CameraParamIndex.K1].unsqueeze(1)  # [B, 1]
        k2 = calib_params[:, CameraParamIndex.K2].unsqueeze(1)  # [B, 1]
        k3 = calib_params[:, CameraParamIndex.K3].unsqueeze(1)  # [B, 1]
        k4 = calib_params[:, CameraParamIndex.K4].unsqueeze(1)  # [B, 1]
        p1 = calib_params[:, CameraParamIndex.P1].unsqueeze(1)  # [B, 1]
        p2 = calib_params[:, CameraParamIndex.P2].unsqueeze(1)  # [B, 1]
        
        # Get image dimensions
        img_width = calib_params[:, CameraParamIndex.IMAGE_WIDTH].unsqueeze(1)  # [B, 1]
        img_height = calib_params[:, CameraParamIndex.IMAGE_HEIGHT].unsqueeze(1)  # [B, 1]
        
        # Get extrinsic parameters (quaternion + translation)
        qw = calib_params[:, CameraParamIndex.QW].unsqueeze(1)  # [B, 1]
        qx = calib_params[:, CameraParamIndex.QX].unsqueeze(1)  # [B, 1]
        qy = calib_params[:, CameraParamIndex.QY].unsqueeze(1)  # [B, 1]
        qz = calib_params[:, CameraParamIndex.QZ].unsqueeze(1)  # [B, 1]
        tx = calib_params[:, CameraParamIndex.X].unsqueeze(1)  # [B, 1]
        ty = calib_params[:, CameraParamIndex.Y].unsqueeze(1)  # [B, 1]
        tz = calib_params[:, CameraParamIndex.Z].unsqueeze(1)  # [B, 1]
        
        # Convert quaternion to rotation matrix
        # Using the quaternion to rotation matrix formula
        r00 = 1 - 2 * (qy * qy + qz * qz)
        r01 = 2 * (qx * qy - qz * qw)
        r02 = 2 * (qx * qz + qy * qw)
        
        r10 = 2 * (qx * qy + qz * qw)
        r11 = 1 - 2 * (qx * qx + qz * qz)
        r12 = 2 * (qy * qz - qx * qw)
        
        r20 = 2 * (qx * qz - qy * qw)
        r21 = 2 * (qy * qz + qx * qw)
        r22 = 1 - 2 * (qx * qx + qy * qy)
        
        # Apply rotation and translation
        x = points_flat[..., 0].unsqueeze(-1)  # [B, N*P, 1]
        y = points_flat[..., 1].unsqueeze(-1)  # [B, N*P, 1]
        z = points_flat[..., 2].unsqueeze(-1)  # [B, N*P, 1]
        # Transform points from ego to camera coordinates
        x_cam = r00 * x + r01 * y + r02 * z + tx
        y_cam = r10 * x + r11 * y + r12 * z + ty
        z_cam = r20 * x + r21 * y + r22 * z + tz
        
        # Check if points are behind the camera
        behind_camera = (z_cam <= 0).squeeze(-1)  # [B, N*P]
        
        # Handle division by zero
        z_cam = torch.where(z_cam == 0, torch.ones_like(z_cam) * 1e-10, z_cam)
        
        # Normalize coordinates
        x_normalized = x_cam / z_cam
        y_normalized = y_cam / z_cam
        
        # Apply camera model based on camera type
        if torch.unique(camera_type).shape[0] == 1:
            # If all cameras are of the same type, avoid branching
            camera_type_value = camera_type[0].item()
            
            if camera_type_value == CameraType.UNKNOWN:
                # Default model with no distortion
                x_distorted = x_normalized
                y_distorted = y_normalized
                
            elif camera_type_value == CameraType.PINHOLE:
                # Standard pinhole camera model with radial and tangential distortion
                r2 = x_normalized * x_normalized + y_normalized * y_normalized
                r4 = r2 * r2
                r6 = r4 * r2
                
                # Radial distortion
                radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
                
                # Tangential distortion
                dx = 2 * p1 * x_normalized * y_normalized + p2 * (r2 + 2 * x_normalized * x_normalized)
                dy = p1 * (r2 + 2 * y_normalized * y_normalized) + 2 * p2 * x_normalized * y_normalized
                
                # Apply distortion
                x_distorted = x_normalized * radial + dx
                y_distorted = y_normalized * radial + dy
                
            elif camera_type_value == CameraType.FISHEYE:
                # Fisheye camera model
                r = torch.sqrt(x_normalized * x_normalized + y_normalized * y_normalized)
                
                # Handle zero radius
                r = torch.where(r == 0, torch.ones_like(r) * 1e-10, r)
                
                # Compute theta (angle from optical axis)
                theta = torch.atan(r)
                theta2 = theta * theta
                theta4 = theta2 * theta2
                theta6 = theta4 * theta2
                theta8 = theta4 * theta4
                
                # Apply distortion model
                theta_d = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
                
                # Scale factors
                scaling = torch.where(r > 0, theta_d / r, torch.ones_like(r))
                
                # Apply scaling
                x_distorted = x_normalized * scaling
                y_distorted = y_normalized * scaling
                
            elif camera_type_value == CameraType.GENERAL_DISTORT:
                # Same as pinhole with distortion
                r2 = x_normalized * x_normalized + y_normalized * y_normalized
                r4 = r2 * r2
                r6 = r4 * r2
                
                # Radial distortion
                radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
                
                # Tangential distortion
                dx = 2 * p1 * x_normalized * y_normalized + p2 * (r2 + 2 * x_normalized * x_normalized)
                dy = p1 * (r2 + 2 * y_normalized * y_normalized) + 2 * p2 * x_normalized * y_normalized
                
                # Apply distortion
                x_distorted = x_normalized * radial + dx
                y_distorted = y_normalized * radial + dy
                
            else:
                # Default to pinhole without distortion
                x_distorted = x_normalized
                y_distorted = y_normalized
        else:
            # Handle mixed camera types in batch (less efficient), should not happen
            x_distorted = torch.zeros_like(x_normalized)
            y_distorted = torch.zeros_like(y_normalized)
            
            for b in range(B):
                if camera_type[b] == CameraType.UNKNOWN:
                    x_distorted[b] = x_normalized[b]
                    y_distorted[b] = y_normalized[b]
                
                elif camera_type[b] == CameraType.PINHOLE:
                    r2 = x_normalized[b] * x_normalized[b] + y_normalized[b] * y_normalized[b]
                    r4 = r2 * r2
                    r6 = r4 * r2
                    
                    radial = 1 + k1[b] * r2 + k2[b] * r4 + k3[b] * r6
                    
                    dx = 2 * p1[b] * x_normalized[b] * y_normalized[b] + p2[b] * (r2 + 2 * x_normalized[b] * x_normalized[b])
                    dy = p1[b] * (r2 + 2 * y_normalized[b] * y_normalized[b]) + 2 * p2[b] * x_normalized[b] * y_normalized[b]
                    
                    x_distorted[b] = x_normalized[b] * radial + dx
                    y_distorted[b] = y_normalized[b] * radial + dy
                
                elif camera_type[b] == CameraType.FISHEYE:
                    r = torch.sqrt(x_normalized[b] * x_normalized[b] + y_normalized[b] * y_normalized[b])
                    r = torch.where(r == 0, torch.ones_like(r) * 1e-10, r)
                    
                    theta = torch.atan(r)
                    theta2 = theta * theta
                    theta4 = theta2 * theta2
                    theta6 = theta4 * theta2
                    theta8 = theta4 * theta4
                    
                    theta_d = theta * (1 + k1[b] * theta2 + k2[b] * theta4 + k3[b] * theta6 + k4[b] * theta8)
                    scaling = torch.where(r > 0, theta_d / r, torch.ones_like(r))
                    
                    x_distorted[b] = x_normalized[b] * scaling
                    y_distorted[b] = y_normalized[b] * scaling
                
                elif camera_type[b] == CameraType.GENERAL_DISTORT:
                    r2 = x_normalized[b] * x_normalized[b] + y_normalized[b] * y_normalized[b]
                    r4 = r2 * r2
                    r6 = r4 * r2
                    
                    radial = 1 + k1[b] * r2 + k2[b] * r4 + k3[b] * r6
                    
                    dx = 2 * p1[b] * x_normalized[b] * y_normalized[b] + p2[b] * (r2 + 2 * x_normalized[b] * x_normalized[b])
                    dy = p1[b] * (r2 + 2 * y_normalized[b] * y_normalized[b]) + 2 * p2[b] * x_normalized[b] * y_normalized[b]
                    
                    x_distorted[b] = x_normalized[b] * radial + dx
                    y_distorted[b] = y_normalized[b] * radial + dy
                
                else:
                    x_distorted[b] = x_normalized[b]
                    y_distorted[b] = y_normalized[b]
        
        # Apply camera matrix
        x_pixel = fx * x_distorted + cx
        y_pixel = fy * y_distorted + cy
        
        # Normalize to [0, 1] for consistency with the visibility check in gather_point_features
        x_norm = x_pixel / img_width
        y_norm = y_pixel / img_height
        
        # Combine coordinates
        points_2d = torch.cat([x_norm, y_norm], dim=-1)
        
        # Reshape back to original dimensions
        points_2d = points_2d.view(B, N, P, 2)
        
        # Mark behind-camera points as invalid (set to a value outside [0,1])
        behind_camera = behind_camera.view(B, N, P)
        points_2d[behind_camera] = -2.0
        
        return points_2d
    