import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F
from base import (
    SourceCameraId, CameraType, ObstacleTrajectory, 
    ObjectType, TrajParamIndex, Point3DAccMotion, AttributeType,
    CameraIntrinsicIndex, ExtrinsicIndex, EgoStateIndex,
    CameraParamIndex
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
    """Trajectory decoder with iterative refinement and feature sampling."""
    def __init__(self,
                 num_layers: int = 6,
                 num_queries: int = 100,
                 feature_dim: int = 256,
                 hidden_dim: int = 512,
                 num_points: int = 25):  # Points to sample per face of the unit cube
        super().__init__()
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.num_points = num_points
        
        # Learnable trajectory queries
        self.trajectory_queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        
        # 1. Motion parameter head - predict normalized 0-1 values for motion & size
        self.motion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, TrajParamIndex.END_OF_INDEX),
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
            queries: Tensor[B, N, hidden_dim]
            
        Returns:
            Dict containing:
                - 'motion_params': Tensor[B, N, TrajParamIndex.END_OF_INDEX]
                - 'object_type': Tensor[B, N] - Class indices
                - 'attributes': Tensor[B, N, AttributeType.END_OF_INDEX]
        """
        B, N = queries.shape[:2]
        
        # 1. Predict motion parameters (normalized 0-1)
        norm_params = self.motion_head(queries)  # [B, N, TrajParamIndex.END_OF_INDEX]
        
        # Scale parameters to real values
        min_vals = self.param_ranges['min']
        max_vals = self.param_ranges['max']
        motion_params = norm_params * (max_vals - min_vals) + min_vals
        
        # 2. Predict object type
        type_logits = self.type_head(queries)  # [B, N, len(ObjectType)]
        type_probs = F.softmax(type_logits, dim=-1)
        object_type = torch.argmax(type_probs, dim=-1)  # [B, N]
        
        # 3. Predict attributes
        attribute_logits = self.attribute_head(queries)  # [B, N, AttributeType.END_OF_INDEX]
        attributes = torch.sigmoid(attribute_logits)  # Each attribute is binary
        
        return {
            'motion_params': motion_params,
            'object_type': object_type,
            'attributes': attributes,
            'type_logits': type_logits
        }
    
    def forward(self, 
                features_dict: Dict[SourceCameraId, torch.Tensor],
                calibrations: Dict[SourceCameraId, Dict],
                ego_states: torch.Tensor) -> Tuple[List[ObstacleTrajectory], List[Dict]]:
        """
        Args:
            features_dict: Dict[camera_id -> Tensor[B, T, C, H, W]]
            calibrations: Dict[camera_id -> calibration_dict]
            ego_states: Tensor[B, T, 5] containing position, yaw, timestamp
            
        Returns:
            - Final trajectories list
            - List of intermediate prediction dicts for auxiliary losses
        """
        B = next(iter(features_dict.values())).shape[0]
        T = next(iter(features_dict.values())).shape[1]
        queries = self.trajectory_queries.expand(B, -1, -1)
        
        all_predictions = []
        
        # Iterative refinement
        for layer_idx, layer in enumerate(self.layers):
            # 1. Predict trajectory parameters from queries
            predictions = self.predict_trajectory_parameters(queries) # [B, N, TrajParamIndex.END_OF_INDEX]
            all_predictions.append(predictions)
            
            # For the last layer, we don't need to update queries
            if layer_idx == self.num_layers - 1:
                break
                
            # 2. Sample points on 3D bounding boxes using motion parameters
            box_points = self.sample_box_points(predictions['motion_params'])  # [B, N, num_points, 3]
            
            # 3. Gather features from all views and frames
            point_features = self.gather_point_features(
                box_points, features_dict, calibrations, ego_states
            )  # [B, N, num_points, T, num_cameras, C]
            
            # 4. Aggregate features
            agg_features = self.aggregate_features(point_features)  # [B, N, hidden_dim]
            
            # 5. Update queries
            queries = layer(queries, agg_features)
        
        # Convert final parameters to ObstacleTrajectory objects
        final_predictions = all_predictions[-1]
        trajectories = self.params_to_trajectories(
            final_predictions['motion_params'][0],  # Use first batch
            final_predictions['object_type'][0],
            final_predictions['attributes'][0]
        )
        
        return trajectories, all_predictions
    
    def params_to_trajectories(self, 
                             motion_params: torch.Tensor, 
                             object_types: torch.Tensor,
                             attributes: torch.Tensor) -> List[ObstacleTrajectory]:
        """Convert predicted parameters to ObstacleTrajectory objects.
        
        Args:
            motion_params: Tensor[N, TrajParamIndex.END_OF_INDEX]
            object_types: Tensor[N] - Object type indices
            attributes: Tensor[N, AttributeType.END_OF_INDEX] - Attribute probabilities
            
        Returns:
            List[ObstacleTrajectory]
        """
        trajectories = []
        
        # Extract has_object probabilities
        has_object_probs = attributes[:, AttributeType.HAS_OBJECT]
        is_static_probs = attributes[:, AttributeType.STATIC]
        is_occluded_probs = attributes[:, AttributeType.OCCLUDED]
        
        # Filter out detections with low confidence
        valid_mask = has_object_probs > 0.5
        
        for i in range(motion_params.shape[0]):
            if not valid_mask[i]:
                continue
                
            p = motion_params[i]
            
            # Create Point3DAccMotion
            motion = Point3DAccMotion(
                x=p[TrajParamIndex.X].item(),
                y=p[TrajParamIndex.Y].item(),
                z=p[TrajParamIndex.Z].item(),
                vx=p[TrajParamIndex.VX].item(),
                vy=p[TrajParamIndex.VY].item(),
                vz=0.0,
                ax=p[TrajParamIndex.AX].item(),
                ay=p[TrajParamIndex.AY].item(),
                az=0.0
            )
            
            # Get static flag from attributes
            is_static = bool(is_static_probs[i].item() > 0.5)
            
            # Create ObstacleTrajectory
            traj = ObstacleTrajectory(
                id=i,
                motion=motion,
                yaw=p[TrajParamIndex.YAW].item(),
                length=p[TrajParamIndex.LENGTH].item(),
                width=p[TrajParamIndex.WIDTH].item(),
                height=p[TrajParamIndex.HEIGHT].item(),
                object_type=ObjectType(object_types[i].item()),
                static=is_static,
                valid=True
            )
            
            trajectories.append(traj)
        
        return trajectories
    
    def sample_box_points(self, motion_params: torch.Tensor) -> torch.Tensor:
        """Sample points on 3D bounding box in object's local coordinate system.
        
        Args:
            motion_params: Tensor[B, N, TrajParamIndex.END_OF_INDEX]
                B: batch size
                N: number of queries
            
        Returns:
            Tensor[B, N, num_points, 3]: Points on box surfaces in object local coordinates
        """
        B, N = motion_params.shape[:2]
        device = motion_params.device
        
        # Get dimensions
        length = motion_params[..., TrajParamIndex.LENGTH].unsqueeze(-1)  # [B, N, 1]
        width = motion_params[..., TrajParamIndex.WIDTH].unsqueeze(-1)    # [B, N, 1]
        height = motion_params[..., TrajParamIndex.HEIGHT].unsqueeze(-1)  # [B, N, 1]
        
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
                            features_dict: Dict[SourceCameraId, torch.Tensor],
                            calibrations: Dict[SourceCameraId, torch.Tensor],
                            ego_states: torch.Tensor) -> torch.Tensor:
        """Gather features for box points from all cameras and frames.
        
        Args:
            box_points: Tensor[B, N, num_points, 3]: Points in object local coordinates
            features_dict: Dict[camera_id -> Tensor[B, T, C, H, W]]
            calibrations: Dict[camera_id -> Tensor[B, CameraParamIndex.END_OF_INDEX]]
            ego_states: Tensor[B, T, EgoStateIndex.END_OF_INDEX]
            
        Returns:
            Tensor[B, N, num_points, T, num_cameras, C]
        """
        B, N, P = box_points.shape[:3]
        T = next(iter(features_dict.values())).shape[1]
        C = next(iter(features_dict.values())).shape[2]
        device = box_points.device
        num_cameras = len(features_dict)
        
        # Store motion parameters for calculations
        params = self.motion_params
        
        # Initialize output tensor to store all features
        all_features = torch.zeros(B, N, P, T, num_cameras, C, device=device)
        
        # For each time step
        for t in range(T):
            # 1. Calculate object positions at time t
            dt = ego_states[:, t, 4] - ego_states[:, -1, 4]  # Time diff from reference frame
            dt = dt.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            
            # Calculate new center position using motion model (constant acceleration)
            # x(t) = x0 + v0*t + 0.5*a*t^2
            pos_x = params[..., TrajParamIndex.X].unsqueeze(-1) + \
                   params[..., TrajParamIndex.VX].unsqueeze(-1) * dt + \
                   0.5 * params[..., TrajParamIndex.AX].unsqueeze(-1) * dt * dt
                   
            pos_y = params[..., TrajParamIndex.Y].unsqueeze(-1) + \
                   params[..., TrajParamIndex.VY].unsqueeze(-1) * dt + \
                   0.5 * params[..., TrajParamIndex.AY].unsqueeze(-1) * dt * dt
                   
            pos_z = params[..., TrajParamIndex.Z].unsqueeze(-1)  # Assume constant z
            
            # Calculate velocity at time t
            # v(t) = v0 + a*t
            vel_x = params[..., TrajParamIndex.VX].unsqueeze(-1) + \
                   params[..., TrajParamIndex.AX].unsqueeze(-1) * dt
                   
            vel_y = params[..., TrajParamIndex.VY].unsqueeze(-1) + \
                   params[..., TrajParamIndex.AY].unsqueeze(-1) * dt
            
            # 2. Determine object orientation
            # Use velocity direction if speed is above threshold, otherwise use predicted yaw
            speed = torch.sqrt(vel_x*vel_x + vel_y*vel_y)
            base_yaw = params[..., TrajParamIndex.YAW].unsqueeze(-1)
            
            # Calculate yaw from velocity when speed is sufficient
            vel_yaw = torch.atan2(vel_y, vel_x)
            
            # Select yaw based on speed
            threshold = 0.1
            yaw = torch.where(speed > threshold, vel_yaw, base_yaw)
            
            # 3. Apply local-to-global transform to box points
            # Rotation matrix (2D rotation around z-axis)
            cos_yaw = torch.cos(yaw)
            sin_yaw = torch.sin(yaw)
            
            # Apply rotation to all points
            local_points = box_points.clone()  # [B, N, P, 3]
            
            # Rotate points
            rotated_x = local_points[..., 0] * cos_yaw - local_points[..., 1] * sin_yaw
            rotated_y = local_points[..., 0] * sin_yaw + local_points[..., 1] * cos_yaw
            rotated_z = local_points[..., 2]
            
            # Concatenate rotated coordinates
            rotated_points = torch.stack([rotated_x, rotated_y, rotated_z], dim=-1)
            
            # Translate to object center
            global_center = torch.cat([pos_x, pos_y, pos_z], dim=-1)  # [B, N, 3]
            global_points = rotated_points + global_center.unsqueeze(2)  # [B, N, P, 3]
            
            # 4. Transform to ego coordinate system at time t
            ego_points = self.transform_points_to_ego_frame(
                global_points, ego_states[:, t]
            )  # [B, N, P, 3]
            
            # 5. Project points to each camera and sample features
            for camera_idx, (camera_id, features) in enumerate(features_dict.items()):
                # Get camera features for this timestep
                features_t = features[:, t]  # [B, C, H, W]
                feature_H, feature_W = features_t.shape[2:4]
                
                # Project points to camera view
                calib = calibrations[camera_id]  # [B, CameraParamIndex.END_OF_INDEX]
                
                # Use first batch's calibration for image dimensions
                img_width = calib[0, CameraParamIndex.IMAGE_WIDTH].item()
                img_height = calib[0, CameraParamIndex.IMAGE_HEIGHT].item()
                
                # Process each batch separately since the calibration can be different
                for b in range(B):
                    # Project 3D points to image coordinates
                    points_2d_b = self.project_points_to_image(
                        ego_points[b].reshape(N*P, 3),
                        calib[b],
                        None,  # extrinsic is now merged into calib tensor
                        (img_width, img_height)
                    )  # [N*P, 2] - normalized coordinates (0-1)
                    
                    # Convert from 0-1 to [-1, 1] for grid_sample
                    points_2d_grid = points_2d_b.clone()
                    points_2d_grid[:, 0] = 2 * points_2d_grid[:, 0] - 1
                    points_2d_grid[:, 1] = 2 * points_2d_grid[:, 1] - 1
                    
                    # Check if points are in image bounds
                    valid_mask = (
                        (points_2d_grid[:, 0] >= -1) & (points_2d_grid[:, 0] <= 1) &
                        (points_2d_grid[:, 1] >= -1) & (points_2d_grid[:, 1] <= 1)
                    ).float().reshape(N, P, 1)
                    
                    # Clamp values to prevent sampling outside
                    points_2d_grid = torch.clamp(points_2d_grid, -1, 1)
                    
                    # Sample features (for single batch)
                    sampled_feats = F.grid_sample(
                        features_t[b:b+1],
                        points_2d_grid.reshape(1, N*P, 1, 2),
                        mode='bilinear',
                        align_corners=False
                    )  # [1, C, N*P, 1]
                    
                    # Reshape and apply valid mask
                    sampled_feats = sampled_feats.reshape(C, N, P).permute(1, 2, 0)  # [N, P, C]
                    sampled_feats = sampled_feats * valid_mask
                    
                    # Store in output tensor
                    all_features[b, :, :, t, camera_idx] = sampled_feats
        
        return all_features
    
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

    def project_points_to_image(self, 
                         points_3d: torch.Tensor, 
                         intrinsic: torch.Tensor, 
                         extrinsic: torch.Tensor,
                         image_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Project 3D points to image coordinates considering different camera types.
        
        Args:
            points_3d: Tensor[N, 3] - 3D points in ego coordinate system
            intrinsic: Tensor[CameraParamIndex.END_OF_INDEX] - Camera parameters
            extrinsic: Unused - kept for backward compatibility
            image_size: Optional tuple (width, height) - If not provided, uses from intrinsic
            
        Returns:
            Tensor[N, 2] - 2D pixel coordinates normalized by image dimensions (0 to 1)
        """
        device = points_3d.device
        
        # Extract camera parameters
        camera_type = int(intrinsic[CameraParamIndex.CAMERA_TYPE].item())
        img_width = intrinsic[CameraParamIndex.IMAGE_WIDTH].item()
        img_height = intrinsic[CameraParamIndex.IMAGE_HEIGHT].item()
        fx = intrinsic[CameraParamIndex.FX]
        fy = intrinsic[CameraParamIndex.FY]
        cx = intrinsic[CameraParamIndex.CX]
        cy = intrinsic[CameraParamIndex.CY]
        
        # Override image size if provided
        if image_size is not None:
            img_width, img_height = image_size
        
        # Distortion parameters
        k1 = intrinsic[CameraParamIndex.K1]
        k2 = intrinsic[CameraParamIndex.K2]
        k3 = intrinsic[CameraParamIndex.K3]
        k4 = intrinsic[CameraParamIndex.K4]
        p1 = intrinsic[CameraParamIndex.P1]
        p2 = intrinsic[CameraParamIndex.P2]
        
        # Convert quaternion to rotation matrix
        qw = intrinsic[CameraParamIndex.QW]
        qx = intrinsic[CameraParamIndex.QX]
        qy = intrinsic[CameraParamIndex.QY]
        qz = intrinsic[CameraParamIndex.QZ]
        
        # Quaternion to rotation matrix
        R = torch.zeros(3, 3, device=device)
        R[0, 0] = 1 - 2*qy*qy - 2*qz*qz
        R[0, 1] = 2*qx*qy - 2*qz*qw
        R[0, 2] = 2*qx*qz + 2*qy*qw
        R[1, 0] = 2*qx*qy + 2*qz*qw
        R[1, 1] = 1 - 2*qx*qx - 2*qz*qz
        R[1, 2] = 2*qy*qz - 2*qx*qw
        R[2, 0] = 2*qx*qz - 2*qy*qw
        R[2, 1] = 2*qy*qz + 2*qx*qw
        R[2, 2] = 1 - 2*qx*qx - 2*qy*qy
        
        # Translation vector
        t = torch.tensor([
            intrinsic[CameraParamIndex.X], 
            intrinsic[CameraParamIndex.Y], 
            intrinsic[CameraParamIndex.Z]
        ], device=device)
        
        # Transform points from ego to camera coordinate system
        # R_cw @ (p - t)
        points_cam = torch.matmul(points_3d - t, R.T)
        
        # Points behind the camera
        behind_camera = (points_cam[:, 2] <= 0)
        
        # Normalized coordinates (x/z, y/z)
        # Handle division by zero
        z = points_cam[:, 2:3]
        z = torch.where(z == 0, torch.ones_like(z) * 1e-10, z)
        x_normalized = points_cam[:, 0:1] / z
        y_normalized = points_cam[:, 1:2] / z
        
        # Apply different camera models based on camera type
        if camera_type == CameraType.PINHOLE:
            # Ideal pinhole camera model without distortion
            x_distorted = x_normalized
            y_distorted = y_normalized
            
        elif camera_type == CameraType.GENERAL_DISTORT:
            # Standard pinhole camera model with radial and tangential distortion
            r2 = x_normalized * x_normalized + y_normalized * y_normalized
            r4 = r2 * r2
            r6 = r4 * r2
            
            # Radial distortion
            radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
            
            # Tangential distortion
            dx_tangential = 2 * p1 * x_normalized * y_normalized + p2 * (r2 + 2 * x_normalized * x_normalized)
            dy_tangential = p1 * (r2 + 2 * y_normalized * y_normalized) + 2 * p2 * x_normalized * y_normalized
            
            # Apply distortion
            x_distorted = x_normalized * radial + dx_tangential
            y_distorted = y_normalized * radial + dy_tangential
            
        elif camera_type == CameraType.FISHEYE:
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
            
        else:
            # Default to pinhole without distortion
            x_distorted = x_normalized
            y_distorted = y_normalized
        
        # Apply camera matrix
        x_pixel = fx * x_distorted + cx
        y_pixel = fy * y_distorted + cy
        
        # Normalize to [0, 1]
        x_norm = x_pixel / img_width
        y_norm = y_pixel / img_height
        
        # Combine coordinates
        points_2d = torch.cat([x_norm, y_norm], dim=1)
        
        # Mark behind-camera points as invalid
        points_2d[behind_camera] = -2.0  # Outside of normalized range
        
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
    
    def forward(self, batch: Dict) -> Dict:
        """Forward pass returning trajectories and auxiliary outputs for training.
        
        Args:
            batch: Dict containing:
                - images: Dict[camera_id -> Tensor[B, T, 3, H, W]]
                - calibrations: Dict[camera_id -> Tensor[B, CameraParamIndex.END_OF_INDEX]]
                - ego_states: Tensor[B, T, EgoStateIndex.END_OF_INDEX]
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
        
        # Decode trajectories using fused features from all cameras
        trajectories, aux_trajectories = self.trajectory_decoder(
            features,
            batch['calibrations'],
            batch['ego_states']
        )
        
        return {
            'trajectories': trajectories,
            'aux_trajectories': aux_trajectories
        }