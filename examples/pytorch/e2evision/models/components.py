import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional, List
import numpy as np
import torch.nn.functional as F

from base import SourceCameraId, TrajParamIndex, CameraParamIndex, EgoStateIndex, CameraType

class ImageFeatureExtractor(nn.Module):
    """Image feature extraction module."""
    def __init__(self, out_channels: int = 256, use_pretrained: bool = False, backbone: str = 'resnet50'):
        super().__init__()
        
        # 选择backbone
        if backbone == 'resnet18':
            # 使用更轻量级的ResNet18
            if use_pretrained:
                resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                resnet = models.resnet18(weights=None)
            self.channel_adjust = nn.Conv2d(512, out_channels, 1)  # ResNet18输出512通道
        elif backbone == 'resnet34':
            if use_pretrained:
                resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            else:
                resnet = models.resnet34(weights=None)
            self.channel_adjust = nn.Conv2d(512, out_channels, 1)  # ResNet34输出512通道
        else:  # 默认使用ResNet50
            # Use ResNet50 as backbone with weights parameter based on use_pretrained flag
            if use_pretrained:
                resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                # Skip downloading weights if not using pretrained
                resnet = models.resnet50(weights=None)
            self.channel_adjust = nn.Conv2d(2048, out_channels, 1)  # ResNet50输出2048通道
        
        # Remove classification head
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Features [B, out_channels, H/32, W/32]
        """
        x = self.backbone(x)
        x = self.channel_adjust(x)
        return x


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
        self.register_buffer('unit_points', self._generate_unit_cube_points(num_points))
        
        # Parameter ranges for normalization: torch.Tensor[TrajParamIndex.HEIGHT + 1, 2]
        ranges = self._get_motion_param_range()
        self.register_buffer('motion_min_vals', ranges[:, 0])
        self.register_buffer('motion_ranges', ranges[:, 1] - ranges[:, 0])
         
        # Single trajectory parameter head that outputs all trajectory parameters
        self.traj_head = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, TrajParamIndex.END_OF_INDEX)
        )
        
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Refiner layers
        self.layers = nn.ModuleList([
            TrajectoryQueryRefineLayer(
                feature_dim=feature_dim,
                num_heads=8,
                dim_feedforward=hidden_dim
            ) for _ in range(num_layers)
        ])
        
        # Initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
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
            traj_params = self.traj_head(queries) 
            # Dict[traj_params: Tensor[B, N, TrajParamIndex.END_OF_INDEX], type_logits: Tensor[B, N, len(ObjectType)]]
            outputs.append(traj_params)
            
            # 2. Sample points on predicted objects
            box_points = self.sample_box_points(traj_params)  # [B, N, num_points, 3]
            
            # 3. Gather features from all views and frames
            point_features = self.gather_point_features(
                box_points, traj_params, features_dict, calibrations, ego_states
            )  # [B, N, num_points, T, num_cameras, C]
            
            # 4. Aggregate features
            agg_features = self.aggregate_features(point_features)  # [B, N, hidden_dim]
            
            # 5. Update queries
            queries = layer(queries, agg_features)
            
            
        # Get final predictions
        outputs.append(self.traj_head(queries))
        
        return outputs
    
    def _generate_unit_cube_points(self, num_points: int = 25):
        """Generate sample points on faces of unit cube.
        
        Args:
            num_points: Number of points to sample on each face
            
        Returns:
            Tensor of shape [num_points*6, 3] containing sampled points
        """
        points = []
        points_per_face = int(np.sqrt(num_points))  # e.g., 5 for 25 points per face
        
        # Sample points on each face
        for dim in range(3):  # x, y, z
            for sign in [-1, 1]:  # negative and positive faces
                # Create grid on face
                if dim == 0:  # yz plane
                    y = torch.linspace(-1, 1, points_per_face)
                    z = torch.linspace(-1, 1, points_per_face)
                    grid_y, grid_z = torch.meshgrid(y, z, indexing='ij')
                    x = torch.full_like(grid_y, sign)
                    points.append(torch.stack([x, grid_y, grid_z], dim=-1))
                    
                elif dim == 1:  # xz plane
                    x = torch.linspace(-1, 1, points_per_face)
                    z = torch.linspace(-1, 1, points_per_face)
                    grid_x, grid_z = torch.meshgrid(x, z, indexing='ij')
                    y = torch.full_like(grid_x, sign)
                    points.append(torch.stack([grid_x, y, grid_z], dim=-1))
                    
                else:  # xy plane
                    x = torch.linspace(-1, 1, points_per_face)
                    y = torch.linspace(-1, 1, points_per_face)
                    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
                    z = torch.full_like(grid_x, sign)
                    points.append(torch.stack([grid_x, grid_y, z], dim=-1))
        
        points = torch.cat([p.reshape(-1, 3) for p in points], dim=0)
        
        return points
    
    def _get_motion_param_range(self)->torch.Tensor:
        """Get parameter ranges for normalization.
        
        Returns:
            Tensor of shape [TrajParamIndex.HEIGHT + 1, 2] containing min/max values
        """
        param_range = torch.zeros(TrajParamIndex.HEIGHT + 1, 2)
        
        # Position ranges (in meters)
        param_range[TrajParamIndex.X] = torch.tensor([-80.0, 160.0])
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
    
    def sample_box_points(self, traj_params: torch.Tensor) -> torch.Tensor:
        """Sample points on 3D bounding box in object's local coordinate system.
        
        Args:
            traj_params: Tensor[B, N, TrajParamIndex.END_OF_INDEX]
        
        Returns:
            Tensor[B, N, num_points, 3]: Points on box surfaces in object local coordinates
        """
        
        # Get dimensions
        length = traj_params[..., TrajParamIndex.LENGTH].unsqueeze(-1)  # [B, N, 1]
        width = traj_params[..., TrajParamIndex.WIDTH].unsqueeze(-1)    # [B, N, 1]
        height = traj_params[..., TrajParamIndex.HEIGHT].unsqueeze(-1)  # [B, N, 1]
        
        # Scale unit cube points by dimensions
        # unit_points: [num_points, 3]
        # dims: [B, N, 3]
        dims = torch.stack([length, width, height], dim=-1)  # [B, N, 1, 3]
         
        # Scale unit cube points by dimensions and ensure same device
        box_points = self.unit_points.unsqueeze(0).unsqueeze(0) * dims  # [B, N, num_points, 3]
        return box_points
    
    def gather_point_features(self, 
                            box_points: torch.Tensor, 
                            traj_params: torch.Tensor,
                            features_dict: Dict[SourceCameraId, torch.Tensor],
                            calibrations: Dict[SourceCameraId, torch.Tensor],
                            ego_states: torch.Tensor) -> torch.Tensor:
        """Gather features for box points from all cameras and frames.
        
        Args:
            box_points: Tensor[B, N, num_points, 3]: Points in object local coordinates
            traj_params: Tensor[B, N, TrajParamIndex.END_OF_INDEX]
            features_dict: Dict[camera_id -> Tensor[B, C, H, W]]
            calibrations: Dict[camera_id -> Tensor[B, CameraParamIndex.END_OF_INDEX]]
            ego_states: Tensor[B, T, EgoStateIndex.END_OF_INDEX]
            
        Returns:
            Tensor[B, N, num_points, T, num_cameras, C]
        """
        B, N, P = box_points.shape[:3]
        T = ego_states.shape[1]
        num_cameras = len(features_dict)
        
        # Initialize output tensor to store all features
        C = next(iter(features_dict.values())).shape[1]
        all_point_features = torch.zeros(B, N, P, T, num_cameras, C).to(box_points.device)
        
        # For each time step, transform points to global frame, then project to cameras
        for t in range(T):
            # 1. Calculate object positions at time t
            dt = ego_states[:, t, EgoStateIndex.TIMESTAMP] - ego_states[:, -1, EgoStateIndex.TIMESTAMP]  # Time diff from reference frame
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
                
                # Sample features
                sampled = F.grid_sample(
                    features, grid, mode='bilinear', 
                    padding_mode='zeros', align_corners=True
                )  # [B, C, N*P, 1]
                
                # Reshape back
                point_features = sampled.permute(0, 2, 3, 1).view(B, N, P, C)
                
                # Set features for invisible points to zero
                point_features = point_features * visible.unsqueeze(-1).float()
                
                # Store in output tensor
                all_point_features[:, :, :, t, cam_idx] = point_features
        
        return all_point_features
        
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
            ego_state: Tensor[B, 5] with position, yaw, timestamp
            
        Returns:
            Tensor[B, N, P, 3]
        """
        B, N, P = points.shape[:3]
        
        # Extract ego position and yaw
        ego_pos = ego_state[..., :3].unsqueeze(1).unsqueeze(2)  # [B, 1, 1, 3]
        ego_yaw = ego_state[..., 3].unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
        
        # Create rotation matrix
        cos_yaw = torch.cos(ego_yaw)
        sin_yaw = torch.sin(ego_yaw)
        
        # Apply inverse transform (ego->world to world->ego)
        # First translate
        translated = points - ego_pos
        
        # Then rotate (inverse rotation)
        x_rotated = translated[..., 0] * cos_yaw + translated[..., 1] * sin_yaw
        y_rotated = -translated[..., 0] * sin_yaw + translated[..., 1] * cos_yaw
        
        transformed = torch.stack(
            [x_rotated, y_rotated, translated[..., 2]], 
            dim=-1
        )
        
        return transformed
    
    def aggregate_features(self, point_features: torch.Tensor) -> torch.Tensor:
        """Aggregate features from all points, frames and cameras.
        
        Args:
            point_features: Tensor[B, N, P, T, num_cameras, C]
            
        Returns:
            Tensor[B, N, hidden_dim]
        """
        B, N = point_features.shape[:2]
        
        # Reshape for processing
        features_flat = point_features.flatten(2, 4)  # [B, N, P*T*num_cameras, C]
        
        # Max pooling over all points, frames and cameras
        pooled_features, _ = torch.max(features_flat, dim=2)  # [B, N, C]
        
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
            # Handle mixed camera types in batch (less efficient)
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
    