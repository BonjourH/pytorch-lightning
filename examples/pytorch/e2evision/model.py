import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Tuple
import torch.nn.functional as F
from base import (
    SourceCameraId, CameraType, ObstacleTrajectory, 
    ObjectType, TrajParamIndex, Point3DAccMotion
)
import numpy as np

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

class MultiCameraFeatureFusion(nn.Module):
    """Fuse features from multiple cameras."""
    def __init__(self, 
                 camera_ids: List[SourceCameraId],
                 feature_dim: int = 256):
        super().__init__()
        self.camera_ids = camera_ids
        self.feature_dim = feature_dim
        
        # Add attention layers for feature fusion
        self.camera_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )
        
    def forward(self, 
                features: Dict[SourceCameraId, torch.Tensor],
                calibrations: Dict[SourceCameraId, Dict]) -> torch.Tensor:
        """
        Args:
            features: Dict[camera_id -> Tensor[B, T, C, H, W]]
            calibrations: Dict[camera_id -> calibration_dict]
        Returns:
            Tensor[B, T, C, H, W]: Fused features
        """
        B, T, C, H, W = next(iter(features.values())).shape
        
        # Collect features from all cameras
        all_features = []
        for camera_id in self.camera_ids:
            feat = features[camera_id]
            # Add camera position encoding
            calib = calibrations[camera_id]
            pos_encoding = self._get_position_encoding(calib['extrinsic'])
            feat = feat + pos_encoding.view(1, 1, -1, 1, 1)
            all_features.append(feat)
            
        # Stack features: [B, T, num_cameras, C, H, W]
        stacked_features = torch.stack(all_features, dim=2)
        
        # Reshape for attention: [B*T, num_cameras, C*H*W]
        feat_flat = stacked_features.flatten(3).view(B*T, len(self.camera_ids), -1)
        
        # Apply attention
        fused_feat, _ = self.camera_attention(feat_flat, feat_flat, feat_flat)
        
        # Reshape back: [B, T, C, H, W]
        fused_feat = fused_feat.view(B, T, C, H, W)
        
        return fused_feat
    
    def _get_position_encoding(self, extrinsic: torch.Tensor) -> torch.Tensor:
        """Generate position encoding from camera extrinsic matrix."""
        # Use translation part as position encoding
        pos = extrinsic[:3, 3]
        return pos

class TrajectoryDecoder(nn.Module):
    """Trajectory decoder with multi-layer refinement, similar to DETR decoder."""
    def __init__(self,
                 num_layers: int = 6,
                 num_queries: int = 100,  # Number of trajectory queries
                 feature_dim: int = 256,
                 num_heads: int = 8,
                 hidden_dim: int = 512):
        super().__init__()
        self.num_queries = num_queries
        self.num_layers = num_layers
        
        # Learnable trajectory queries
        self.trajectory_queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        
        # Self-attention and cross-attention layers
        self.layers = nn.ModuleList([
            TrajectoryDecoderLayer(
                hidden_dim=hidden_dim,
                feature_dim=feature_dim,
                num_heads=num_heads
            ) for _ in range(num_layers)
        ])
        
        # Final MLP to predict trajectory parameters
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, MotionParam.END_OF_INDEX)
        )

    def forward(self, features_dict: Dict[SourceCameraId, torch.Tensor],
                calibrations: Dict[SourceCameraId, Dict]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            features_dict: Dict[camera_id -> Tensor[B, T, C, H, W]]
            calibrations: Dict[camera_id -> calibration_dict]
        Returns:
            - Final trajectory parameters [B, num_queries, MotionParam.END_OF_INDEX]
            - Intermediate trajectory parameters for auxiliary losses
        """
        B = next(iter(features_dict.values())).shape[0]
        queries = self.trajectory_queries.expand(B, -1, -1)
        
        intermediate_trajs = []
        
        # Iteratively refine trajectories
        for layer in self.layers:
            # Convert queries to trajectory parameters
            traj_params = self.trajectory_head(queries)
            intermediate_trajs.append(traj_params)
            
            # Sample image features using current trajectory
            sampled_features = self.sample_trajectory_features(
                traj_params, features_dict, calibrations
            )
            
            # Update queries through self/cross attention
            queries = layer(queries, sampled_features)
        
        # Final trajectory prediction
        final_trajs = self.trajectory_head(queries)
        
        return final_trajs, intermediate_trajs

    def sample_trajectory_features(self, 
                                traj_params: torch.Tensor,
                                features_dict: Dict[SourceCameraId, torch.Tensor],
                                calibrations: Dict[SourceCameraId, Dict]) -> torch.Tensor:
        """Sample features for each trajectory from all cameras and frames."""
        B, N = traj_params.shape[:2]  # batch_size, num_queries
        all_features = []
        
        for camera_id, features in features_dict.items():
            # Get keypoints for each trajectory
            keypoints = self.get_trajectory_keypoints(traj_params, 0.0)  # [B, N, 8, 3]
            
            # Project to image plane for each time step
            for t in range(features.shape[1]):  # Iterate over time steps
                points_2d = self.project_points_to_image(
                    keypoints.flatten(0, 1),  # [B*N, 8, 3]
                    calibrations[camera_id]['intrinsic'],
                    calibrations[camera_id]['extrinsic']
                )  # [B*N, 8, 2]
                
                # Sample features for each keypoint
                B, T, C, H, W = features.shape
                features_flat = features[:, t].flatten(0, 1)  # [B, C, H, W]
                
                # Normalize coordinates to [-1, 1]
                points_2d[..., 0] = points_2d[..., 0] / (W / 2) - 1
                points_2d[..., 1] = points_2d[..., 1] / (H / 2) - 1
                
                sampled_feat = F.grid_sample(
                    features_flat,
                    points_2d.view(B*N, 8, 1, 2),
                    mode='bilinear',
                    align_corners=True
                )  # [B*N, C, 8, 1]
                
                all_features.append(sampled_feat.squeeze(-1))
        
        # Aggregate features across cameras, keypoints, and time
        features_cat = torch.cat(all_features, dim=1)  # [B*N, C*num_cameras*T, 8]
        return features_cat.view(B, N, -1)  # [B, N, C*num_cameras*8*T]

    def params_to_trajectory(self, params: torch.Tensor) -> List[ObstacleTrajectory]:
        """Convert network output parameters to ObstacleTrajectory objects."""
        trajectories = []
        for p in params:
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
            traj = ObstacleTrajectory(
                id=-1,  # Assign temporary ID
                t0=0,
                motion=motion,
                yaw=p[TrajParamIndex.YAW].item(),
                length=p[TrajParamIndex.LENGTH].item(),
                width=p[TrajParamIndex.WIDTH].item(),
                height=p[TrajParamIndex.HEIGHT].item(),
                object_type=ObjectType(int(p[TrajParamIndex.TYPE].item())),
                valid=True
            )
            trajectories.append(traj)
        return trajectories

    def get_trajectory_keypoints(self, traj_params: torch.Tensor, timestamp: float) -> torch.Tensor:
        """Generate keypoints using Trajectory class."""
        trajectories = self.params_to_trajectory(traj_params)
        corners = torch.stack([
            torch.from_numpy(traj.corners(timestamp)).float()
            for traj in trajectories
        ])
        return corners

    def project_points_to_image(self, 
                              points_3d: torch.Tensor,
                              intrinsic: torch.Tensor,
                              extrinsic: torch.Tensor) -> torch.Tensor:
        """Project 3D points to image plane."""
        # points_3d: [B, N, 3]
        # Add ones for homogeneous coordinates
        points_h = torch.cat([points_3d, torch.ones_like(points_3d[...,:1])], dim=-1)
        
        # Transform to camera coordinates
        points_cam = torch.bmm(points_h, extrinsic.transpose(1, 2))
        
        # Project to image plane
        points_img = torch.bmm(points_cam[...,:3], intrinsic.transpose(1, 2))
        
        # Convert to 2D coordinates
        points_2d = points_img[...,:2] / points_img[...,[2]]
        return points_2d

class TrajectoryDecoderLayer(nn.Module):
    """Single layer of trajectory decoder."""
    def __init__(self, hidden_dim: int, feature_dim: int, num_heads: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, queries: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # Self attention
        q = self.norm1(queries)
        q = q + self.self_attn(q, q, q)[0]
        
        # Cross attention with sampled features
        q = self.norm2(q)
        features = self.feature_proj(features)
        q = q + self.cross_attn(q, features, features)[0]
        
        # FFN
        q = self.norm3(q)
        q = q + self.mlp(q)
        
        return q

class E2EPerceptionNet(nn.Module):
    def __init__(self,
                 camera_ids: List[SourceCameraId],
                 feature_dim: int = 256,
                 num_queries: int = 100,
                 num_decoder_layers: int = 6):
        super().__init__()
        self.camera_ids = camera_ids
        
        # Image feature extraction
        self.feature_extractor = ImageFeatureExtractor(feature_dim)
        
        # Multi-camera feature fusion
        self.feature_fusion = MultiCameraFeatureFusion(
            camera_ids=camera_ids,
            feature_dim=feature_dim
        )
        
        # Trajectory decoder
        self.trajectory_decoder = TrajectoryDecoder(
            num_layers=num_decoder_layers,
            num_queries=num_queries,
            feature_dim=feature_dim
        )
    
    def forward(self, batch: Dict) -> Dict:
        """Forward pass returning trajectories and auxiliary outputs for training."""
        # Extract features from each camera
        features = {}
        for camera_id in self.camera_ids:
            images = batch['images'][camera_id]
            B, T = images.shape[:2]
            # # batch['images'][SourceCameraId.FRONT_CENTER_CAMERA].shape => torch.Size([2, 10, 3, 416, 800]
            feat = self.feature_extractor(images.flatten(0, 1))
            feat = feat.view(B, T, *feat.shape[1:])
            features[camera_id] = feat
        
        # Fuse features from multiple cameras
        fused_features = self.feature_fusion(features, batch['calibrations'])
        
        # Decode trajectories using fused features
        trajectories, aux_trajectories = self.trajectory_decoder(
            {'fused': fused_features},  # Pass fused features instead of per-camera features
            batch['calibrations']
        )
        
        return {
            'trajectories': trajectories,
            'aux_trajectories': aux_trajectories
        }