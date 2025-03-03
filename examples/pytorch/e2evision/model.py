import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F
from base import (
    SourceCameraId, CameraType, ObstacleTrajectory, 
    ObjectType, TrajParamIndex, Point3DAccMotion, AttributeType,
    EgoStateIndex, CameraParamIndex
)
import numpy as np
from temporal_fusion import TemporalAttentionFusion

class ImageFeatureExtractor(nn.Module):
    """2D CNN backbone for extracting image features."""
    def __init__(self, out_channels: int = 256):
        super().__init__()
        # Use ResNet50 as backbone
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-2],  # Remove avg pool and fc
            nn.Conv2d(2048, out_channels, 1),  # Reduce channel dimension
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    

class TrajectoryDecoder(nn.Module):
    """Trajectory decoder module."""
    
    def __init__(self,
                 num_layers: int = 6,
                 num_queries: int = 100,
                 feature_dim: int = 256,
                 hidden_dim: int = 512,
                 num_points: int = 25):  # Points to sample per face of the unit cube
        super().__init__()
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_points = num_points
        
        # Learnable trajectory queries
        self.trajectory_queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        
        # 1. Motion parameter head - predict normalized 0-1 values for motion & size
        self.motion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, TrajParamIndex.HEIGHT + 1),
            nn.Sigmoid()  # Output normalized 0-1 values
        )
        
        # 2. Object type classification head
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(ObjectType))
        )
        
        # 3. Attribute prediction head
        self.attribute_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, AttributeType.END_OF_INDEX)
        )
        
        # Parameter scaling values (learnable)
        self.param_ranges = nn.ParameterDict({
            'min': nn.Parameter(torch.tensor([
                -50.0, -50.0, -5.0,        # x, y, z position
                -15.0, -15.0,              # vx, vy velocity
                -5.0, -5.0,                # ax, ay acceleration
                -3.14,                     # yaw
                0.5, 0.5, 0.5              # length, width, height
            ])),
            'max': nn.Parameter(torch.tensor([
                50.0, 50.0, 5.0,           # x, y, z position
                15.0, 15.0,                # vx, vy velocity
                5.0, 5.0,                  # ax, ay acceleration
                3.14,                      # yaw
                10.0, 5.0, 3.0             # length, width, height
            ])),
        })
        
        # Feature aggregation MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Query update transformer layers
        self.layers = nn.ModuleList([
            TrajectoryRefinementLayer(
                hidden_dim=hidden_dim,
                num_heads=8
            ) for _ in range(num_layers)
        ])
        
        # Pre-compute unit cube points
        self.unit_cube_points = self._generate_unit_cube_points()
    
    def _generate_unit_cube_points(self):
        """Generate points on unit cube faces at initialization time."""
        # Number of points per edge
        points_per_edge = int(np.sqrt(self.num_points))
        
        # Create unit coordinates (not including 1.0)
        coords = torch.linspace(0, 1, points_per_edge+1)[:-1]  # [0, 1/n, 2/n, ..., (n-1)/n]
        
        # Generate grid points for all faces of the cube
        xx, yy = torch.meshgrid(coords, coords, indexing='ij')
        xx = xx.reshape(-1)  # Flatten
        yy = yy.reshape(-1)  # Flatten
        
        # 6 faces of the unit cube
        all_points = []
        
        # Front face (x=0)
        all_points.append(torch.stack([torch.zeros_like(xx), xx, yy], dim=-1))
        
        # Back face (x=1)
        all_points.append(torch.stack([torch.ones_like(xx), xx, yy], dim=-1))
        
        # Left face (y=0)
        all_points.append(torch.stack([xx, torch.zeros_like(xx), yy], dim=-1))
        
        # Right face (y=1)
        all_points.append(torch.stack([xx, torch.ones_like(xx), yy], dim=-1))
        
        # Bottom face (z=0)
        all_points.append(torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1))
        
        # Top face (z=1)
        all_points.append(torch.stack([xx, yy, torch.ones_like(xx)], dim=-1))
        
        # Combine all faces
        cube_points = torch.cat(all_points, dim=0)  # [num_points, 3]
        
        # Convert from unit cube [0,1] to centered unit cube [-0.5,0.5]
        cube_points = cube_points - 0.5
        
        return cube_points
    
    def predict_trajectory_parameters(self, queries: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict trajectory parameters from query embeddings.
        
        Args:
            queries: Tensor[B, N, hidden_dim] - Object queries
            
        Returns:
            Dict with:
            - traj_params: Tensor[B, N, TrajParamIndex.END_OF_INDEX] - Trajectory parameters
            - type_logits: Tensor[B, N, len(ObjectType)] - Object type logits
        """
        B, N, _ = queries.shape
        device = queries.device
        
        # Initialize trajectory parameter tensor
        traj_params = torch.zeros(B, N, TrajParamIndex.END_OF_INDEX, device=device)
        
        # 1. Predict motion parameters (normalized 0-1)
        norm_params = self.motion_head(queries)  # [B, N, TrajParamIndex.HEIGHT + 1]
        
        # Scale parameters to real values
        min_vals = self.param_ranges['min']
        max_vals = self.param_ranges['max']
        motion_params = norm_params * (max_vals - min_vals) + min_vals
        traj_params[:, :, :TrajParamIndex.HEIGHT + 1] = motion_params
        
        # 2. Predict object type
        type_logits = self.type_head(queries)  # [B, N, len(ObjectType)]
        type_probs = F.softmax(type_logits, dim=-1)
        object_type = torch.argmax(type_probs, dim=-1)  # [B, N]
        traj_params[:, :, TrajParamIndex.OBJECT_TYPE] = object_type.float()
        
        # 3. Predict attributes
        attribute_logits = self.attribute_head(queries)  # [B, N, AttributeType.END_OF_INDEX]
        attributes = torch.sigmoid(attribute_logits)  # Each attribute is binary
        
        # Map attributes to trajectory parameters
        traj_params[:, :, TrajParamIndex.HAS_OBJECT] = attributes[:, :, AttributeType.HAS_OBJECT]
        traj_params[:, :, TrajParamIndex.STATIC] = attributes[:, :, AttributeType.STATIC]
        traj_params[:, :, TrajParamIndex.OCCLUDED] = attributes[:, :, AttributeType.OCCLUDED]
        
        return {
            'traj_params': traj_params, # Tensor[B, N, TrajParamIndex.END_OF_INDEX]
            'type_logits': type_logits # Tensor[B, N, len(ObjectType)]
        }
    
    def forward(self, 
                features_dict: Dict[SourceCameraId, torch.Tensor],
                calibrations: Dict[SourceCameraId, torch.Tensor],
                ego_states: torch.Tensor) -> List[Dict]:
        """Forward pass to predict trajectories from features.
        
        Args:
            features_dict: Dict[camera_id -> Tensor[B, T, C, H, W]]
            calibrations: Dict[camera_id -> Tensor[B, CameraParamIndex.END_OF_INDEX]]
            ego_states: Tensor[B, T, EgoStateIndex.END_OF_INDEX]
            
        Returns:
            List[Dict]
                - traj_params: Tensor[B, N, TrajParamIndex.END_OF_INDEX]
                - type_logits: Tensor[B, N, len(ObjectType)]
        """
        B = next(iter(features_dict.values())).shape[0]
        device = next(iter(features_dict.values())).device
        
        # Initialize object queries
        queries = self.trajectory_queries.repeat(B, 1, 1)  # [B, N, hidden_dim]
        
        # List to store all predictions (for auxiliary loss)
        iterative_predictions = []
        
        # Decoder iteratively refines predictions
        for layer in self.layers:
            # 1. Predict parameters from current queries
            predictions = self.predict_trajectory_parameters(queries) 
            # Dict[traj_params: Tensor[B, N, TrajParamIndex.END_OF_INDEX], type_logits: Tensor[B, N, len(ObjectType)]]
            iterative_predictions.append(predictions)
            
            # 2. Sample points on predicted objects
            box_points = self.sample_box_points(predictions['traj_params'])  # [B, N, num_points, 3]
            
            # 3. Gather features from all views and frames
            point_features = self.gather_point_features(
                box_points, predictions['traj_params'], features_dict, calibrations, ego_states
            )  # [B, N, num_points, T, num_cameras, C]
            
            # 4. Aggregate features
            agg_features = self.aggregate_features(point_features)  # [B, N, hidden_dim]
            
            # 5. Update queries
            queries = layer(queries, agg_features)
        
        # Get final predictions
        iterative_predictions.append(self.predict_trajectory_parameters(queries))  
         
        return iterative_predictions
    
    def sample_box_points(self, traj_params: torch.Tensor) -> torch.Tensor:
        """Sample points on 3D bounding box in object's local coordinate system.
        
        Args:
            traj_params: Tensor[B, N, TrajParamIndex.END_OF_INDEX]
                B: batch size
                N: number of queries
            
        Returns:
            Tensor[B, N, num_points, 3]: Points on box surfaces in object local coordinates
        """
        device = traj_params.device
        
        # Get dimensions
        length = traj_params[..., TrajParamIndex.LENGTH].unsqueeze(-1)  # [B, N, 1]
        width = traj_params[..., TrajParamIndex.WIDTH].unsqueeze(-1)    # [B, N, 1]
        height = traj_params[..., TrajParamIndex.HEIGHT].unsqueeze(-1)  # [B, N, 1]
        
        # Scale unit cube points by dimensions
        # unit_cube_points: [num_points, 3]
        # dims: [B, N, 3]
        dims = torch.stack([length, width, height], dim=-1)  # [B, N, 3]
        
        # Get unit cube points
        unit_points = self.unit_cube_points.to(device)  # [num_points, 3]
        
        # Scale points by dimensions
        box_points = unit_points.unsqueeze(0).unsqueeze(0) * dims.unsqueeze(2)  # [B, N, num_points, 3]
        
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
            features_dict: Dict[camera_id -> Tensor[B, T, C, H, W]]
            calibrations: Dict[camera_id -> Tensor[B, CameraParamIndex.END_OF_INDEX]]
            ego_states: Tensor[B, T, EgoStateIndex.END_OF_INDEX]
            
        Returns:
            Tensor[B, N, num_points, T, num_cameras, C]
        """
        B, N, P = box_points.shape[:3]
        T = next(iter(features_dict.values())).shape[1]
        device = box_points.device
        num_cameras = len(features_dict)
        
        # Initialize output tensor to store all features
        C = next(iter(features_dict.values())).shape[2]  # Feature channels
        all_point_features = torch.zeros(B, N, P, T, num_cameras, C, device=device)
        
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
                frame_features = features[:, t]  # [B, C, H, W]
                H, W = frame_features.shape[-2:]
                
                # Convert normalized coordinates [0,1] to grid coordinates [-1,1] for grid_sample
                norm_points = torch.zeros_like(points_2d)
                norm_points[..., 0] = 2.0 * points_2d[..., 0] - 1.0
                norm_points[..., 1] = 2.0 * points_2d[..., 1] - 1.0
                
                # Reshape for grid_sample
                grid = norm_points.view(B, N * P, 1, 2)
                
                # Sample features
                sampled = F.grid_sample(
                    frame_features, grid, mode='bilinear', 
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
        device = box_points.device
        
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

class TrajectoryRefinementLayer(nn.Module):
    """Layer for refining trajectory queries."""
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Feature attention
        self.feature_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
    def forward(self, queries: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: Tensor[B, N, hidden_dim]
            features: Tensor[B, N, hidden_dim]
            
        Returns:
            Tensor[B, N, hidden_dim]
        """
        # Self-attention
        q = self.norm1(queries)
        q = q + self.self_attn(q, q, q)[0]
        
        # Feature attention
        q = self.norm2(q)
        q = q + self.feature_attn(q, features, features)[0]
        
        # FFN
        q = self.norm3(q)
        q = q + self.ffn(q)
        
        return q

class E2EPerceptionNet(nn.Module):
    """End-to-end multi-camera 3D perception network."""
    def __init__(self,
                 camera_ids: List[SourceCameraId],
                 feature_dim: int = 256,
                 num_queries: int = 100,
                 num_decoder_layers: int = 6):
        super().__init__()
        self.camera_ids = camera_ids
        
        # Image feature extraction
        self.feature_extractor = ImageFeatureExtractor(feature_dim)
        
        # Per-camera temporal fusion
        self.temporal_fusion = nn.ModuleDict({
            str(int(camera_id)): TemporalAttentionFusion(
                feature_dim=feature_dim,
                num_heads=8,
                dropout=0.1
            ) for camera_id in camera_ids
        })
        
        # Trajectory decoder
        self.trajectory_decoder = TrajectoryDecoder(
            num_layers=num_decoder_layers,
            num_queries=num_queries,
            feature_dim=feature_dim
        )
    
    def forward(self, batch: Dict) -> List[Dict]:
        """Forward pass for end-to-end perception.
        
        Args:
            batch: Dict containing:
                - images: Dict[SourceCameraId -> Tensor[B, T, C, H, W]]
                - calibrations: Dict[SourceCameraId -> Tensor[B, CameraParamIndex.END_OF_INDEX]]
                - ego_states: Tensor[B, T, EgoStateIndex.END_OF_INDEX]
                
        Returns:
            iterative_predictions: List[Dict]
                - traj_params: Tensor[B, N, TrajParamIndex.END_OF_INDEX]
                - type_logits: Tensor[B, N, len(ObjectType)]
            
        """
        # Extract features from each camera
        features = {}
        for camera_id in self.camera_ids:
            images = batch['images'][camera_id]  # [B, T, 3, H, W]
            B, T = images.shape[:2]
            
            # Extract features for all frames
            feat = self.feature_extractor(images.flatten(0, 1))  # [B*T, C, H, W]
            feat = feat.view(B, T, *feat.shape[1:])  # [B, T, C, H, W]
            
            # Apply temporal fusion for each camera independently
            fused_feat = self.temporal_fusion[str(int(camera_id))](feat)  # [B, T, C, H, W]
            
            # Store fused features
            features[camera_id] = fused_feat
        
        return self.trajectory_decoder(features, batch['calibrations'], batch['ego_states'])
