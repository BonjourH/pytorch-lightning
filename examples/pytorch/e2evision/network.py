import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Tuple
import numpy as np
import torch.nn.functional as F

class ImageFeatureExtractor(nn.Module):
    """2D backbone for extracting image features."""
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        # Use ResNet50 as backbone
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-2],  # Remove avg pool and fc
            nn.Conv2d(2048, feature_dim, 1),  # Reduce channel dimension
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class TrajectoryNet(nn.Module):
    """Network for trajectory-based multi-camera perception."""
    def __init__(self, 
                 camera_ids: List[str],
                 feature_dim: int = 256,
                 num_keypoints: int = 8,
                 hidden_dim: int = 512):
        super().__init__()
        self.camera_ids = camera_ids
        self.num_keypoints = num_keypoints
        
        # Feature extractors (shared weights across cameras)
        self.feature_extractor = ImageFeatureExtractor(feature_dim)
        
        # MLP for feature aggregation and trajectory optimization
        self.trajectory_mlp = nn.Sequential(
            nn.Linear(feature_dim * num_keypoints * len(camera_ids), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 8)  # [x, y, z, vx, vy, length, width, height]
        )

    def project_points_to_image(self,
                              points_3d: torch.Tensor,
                              intrinsic: torch.Tensor,
                              extrinsic: torch.Tensor) -> torch.Tensor:
        """Project 3D points to 2D image plane.
        
        Args:
            points_3d: [B, N, 3] tensor of 3D points
            intrinsic: [B, 3, 3] camera intrinsic matrix
            extrinsic: [B, 4, 4] camera extrinsic matrix
            
        Returns:
            points_2d: [B, N, 2] tensor of 2D points
        """
        B, N, _ = points_3d.shape
        
        # Homogeneous coordinates
        points_h = torch.cat([points_3d, torch.ones(B, N, 1, device=points_3d.device)], dim=-1)
        
        # World to camera transformation
        points_cam = torch.bmm(points_h, extrinsic.transpose(1, 2))
        
        # Perspective projection
        points_2d = torch.bmm(points_cam[:, :, :3], intrinsic.transpose(1, 2))
        points_2d = points_2d[:, :, :2] / (points_2d[:, :, 2:] + 1e-6)
        
        return points_2d

    def sample_trajectory_keypoints(self,
                                  trajectory_params: torch.Tensor,
                                  timestamps: torch.Tensor) -> torch.Tensor:
        """Sample keypoints from trajectory candidates.
        
        Args:
            trajectory_params: [B, 8] tensor of [x, y, z, vx, vy, length, width, height]
            timestamps: [B, T] tensor of timestamps
            
        Returns:
            keypoints: [B, T, N, 3] tensor of sampled 3D keypoints
        """
        B, T = timestamps.shape
        
        # Extract parameters
        x0, y0, z0 = trajectory_params[:, 0:3].unbind(-1)
        vx, vy = trajectory_params[:, 3:5].unbind(-1)
        l, w, h = trajectory_params[:, 5:8].unbind(-1)
        
        # Compute center positions at each timestamp
        t = timestamps - timestamps[:, 0:1]  # Relative time
        x = x0[:, None] + vx[:, None] * t  # [B, T]
        y = y0[:, None] + vy[:, None] * t  # [B, T]
        z = z0[:, None].expand(-1, T)      # [B, T]
        
        # Sample 8 corner points
        dx = l[:, None, None] / 2  # [B, 1, 1]
        dy = w[:, None, None] / 2
        dz = h[:, None, None] / 2
        
        corners_x = x[:, :, None] + torch.tensor([-1, 1, 1, -1, -1, 1, 1, -1])[None, None, :] * dx
        corners_y = y[:, :, None] + torch.tensor([-1, -1, 1, 1, -1, -1, 1, 1])[None, None, :] * dy
        corners_z = z[:, :, None] + torch.tensor([-1, -1, -1, -1, 1, 1, 1, 1])[None, None, :] * dz
        
        keypoints = torch.stack([corners_x, corners_y, corners_z], dim=-1)  # [B, T, 8, 3]
        return keypoints

    def forward(self, 
                images: Dict[str, torch.Tensor],
                calibrations: Dict[str, Dict[str, torch.Tensor]],
                timestamps: torch.Tensor,
                trajectory_params: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            images: Dict[camera_id -> Tensor[B, T, C, H, W]]
            calibrations: Dict[camera_id -> Dict[intrinsic, extrinsic]]
            timestamps: [B, T] tensor of timestamps
            trajectory_params: [B, 8] initial trajectory parameters
            
        Returns:
            optimized_params: [B, 8] optimized trajectory parameters
        """
        B, T, C, H, W = next(iter(images.values())).shape
        
        # Extract image features for all cameras and timestamps
        features = {}
        for camera_id in self.camera_ids:
            # Reshape for batch processing
            img = images[camera_id].view(B*T, C, H, W)
            feat = self.feature_extractor(img)  # [B*T, C', H', W']
            _, C_, H_, W_ = feat.shape
            features[camera_id] = feat.view(B, T, C_, H_, W_)
        
        # Sample trajectory keypoints
        keypoints = self.sample_trajectory_keypoints(trajectory_params, timestamps)  # [B, T, N, 3]
        
        # Collect features for each keypoint from all cameras
        all_features = []
        for camera_id in self.camera_ids:
            # Project keypoints to image plane
            points_2d = self.project_points_to_image(
                keypoints.view(B*T, self.num_keypoints, 3),
                calibrations[camera_id]['intrinsic'].repeat(T, 1, 1),
                calibrations[camera_id]['extrinsic'].repeat(T, 1, 1)
            )  # [B*T, N, 2]
            
            # Normalize coordinates to [-1, 1] for grid_sample
            points_2d = points_2d.view(B, T, self.num_keypoints, 2)
            points_2d[..., 0] = points_2d[..., 0] / (W_ / 2) - 1
            points_2d[..., 1] = points_2d[..., 1] / (H_ / 2) - 1
            
            # Sample features for each keypoint
            feat = features[camera_id]  # [B, T, C, H, W]
            points_features = F.grid_sample(
                feat.view(B*T, C_, H_, W_),
                points_2d.view(B*T, self.num_keypoints, 1, 2),
                mode='bilinear',
                align_corners=True
            )  # [B*T, C, N, 1]
            
            all_features.append(points_features.squeeze(-1).view(B, T, C_, self.num_keypoints))
        
        # Aggregate features across time and cameras
        features_cat = torch.cat(all_features, dim=-1)  # [B, T, C, N*num_cameras]
        features_mean = features_cat.mean(dim=1)  # [B, C, N*num_cameras]
        
        # Modify the trajectory refinement
        features_flat = features_mean.flatten(1)  # [B, C*N*num_cameras]
        trajectory_update = self.trajectory_mlp(features_flat)
        
        # Apply the update as a residual
        optimized_params = trajectory_params + 0.1 * trajectory_update  # Scale the update
        
        # Ensure positive values for dimensions
        optimized_params[:, 5:] = F.relu(optimized_params[:, 5:])  # length, width, height must be positive
        
        return optimized_params