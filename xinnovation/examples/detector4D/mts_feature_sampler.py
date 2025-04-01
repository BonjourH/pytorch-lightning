import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from xinnovation.src.core.registry import PLUGINS
from xinnovation.src.core import SourceCameraId

__all__ = ["MultiviewTemporalSpatialFeatureSampler"]

@PLUGINS.register_module()
class MultiviewTemporalSpatialFeatureSampler(nn.Module):
    """Sample features from multiple views and temporal frames.
    
    This class samples features from multiple camera views and temporal frames
    for each anchor point in 3D space.
    """
    
    def __init__(self,
                 num_points: int = 8,
                 num_temporal_frames: int = 3,
                 num_spatial_frames: int = 3,
                 **kwargs):
        """Initialize the feature sampler.
        
        Args:
            num_points: Number of points to sample for each anchor
            num_temporal_frames: Number of temporal frames to sample
            num_spatial_frames: Number of spatial frames to sample
        """
        super().__init__()
        self.num_points = num_points
        self.num_temporal_frames = num_temporal_frames
        self.num_spatial_frames = num_spatial_frames
        
    def _sample_temporal_frames(self, 
                              features: torch.Tensor,
                              temporal_indices: torch.Tensor) -> torch.Tensor:
        """Sample features from temporal frames.
        
        Args:
            features: Features with shape [B, T, C, H, W]
            temporal_indices: Indices of temporal frames to sample [B, N, num_temporal_frames]
            
        Returns:
            torch.Tensor: Sampled features with shape [B, N, num_temporal_frames, C, H, W]
        """
        B, T, C, H, W = features.shape
        N = temporal_indices.shape[1]
        
        # Reshape features for sampling
        features = features.unsqueeze(1).expand(-1, N, -1, -1, -1, -1)
        temporal_indices = temporal_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        temporal_indices = temporal_indices.expand(-1, -1, -1, C, H, W)
        
        # Sample features
        sampled_features = torch.gather(features, 2, temporal_indices)
        
        return sampled_features
    
    def _sample_spatial_frames(self,
                             features: torch.Tensor,
                             spatial_indices: torch.Tensor) -> torch.Tensor:
        """Sample features from spatial frames.
        
        Args:
            features: Features with shape [B, N, num_temporal_frames, C, H, W]
            spatial_indices: Indices of spatial frames to sample [B, N, num_spatial_frames]
            
        Returns:
            torch.Tensor: Sampled features with shape [B, N, num_spatial_frames, C, H, W]
        """
        B, N, T, C, H, W = features.shape
        
        # Reshape features for sampling
        features = features.unsqueeze(2).expand(-1, -1, self.num_spatial_frames, -1, -1, -1, -1)
        spatial_indices = spatial_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        spatial_indices = spatial_indices.expand(-1, -1, -1, T, C, H, W)
        
        # Sample features
        sampled_features = torch.gather(features, 3, spatial_indices)
        
        return sampled_features
    
    def _sample_points(self,
                      features: torch.Tensor,
                      point_indices: torch.Tensor) -> torch.Tensor:
        """Sample features at specific points.
        
        Args:
            features: Features with shape [B, N, num_temporal_frames, num_spatial_frames, C, H, W]
            point_indices: Indices of points to sample [B, N, num_points, 2]
            
        Returns:
            torch.Tensor: Sampled features with shape [B, N, num_points, C]
        """
        B, N, T, S, C, H, W = features.shape
        
        # Reshape features for sampling
        features = features.permute(0, 1, 2, 3, 5, 6, 4)  # [B, N, T, S, H, W, C]
        features = features.reshape(B, N, T * S * H * W, C)
        
        # Sample features
        point_indices = point_indices.unsqueeze(-1).expand(-1, -1, -1, C)
        sampled_features = torch.gather(features, 2, point_indices)
        
        return sampled_features
    
    def forward(self,
                features: Dict[SourceCameraId, torch.Tensor],
                anchor_centers: torch.Tensor,
                anchor_corners: torch.Tensor,
                calibrations: Dict[SourceCameraId, torch.Tensor],
                ego_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            features: Dictionary mapping camera IDs to feature maps
            anchor_centers: Anchor centers with shape [B, N, 3]
            anchor_corners: Anchor corners with shape [B, N, 8, 3]
            calibrations: Dictionary mapping camera IDs to calibration parameters
            ego_states: Ego vehicle states with shape [B, T, EgoStateIndex.END_OF_INDEX]
            
        Returns:
            Dict containing:
                - sampled_features: Sampled features with shape [B, N, num_points, C]
                - temporal_indices: Indices of temporal frames sampled
                - spatial_indices: Indices of spatial frames sampled
                - point_indices: Indices of points sampled
        """
        B, N = anchor_centers.shape[:2]
        
        # Generate sampling indices
        temporal_indices = self._generate_temporal_indices(B, N)
        spatial_indices = self._generate_spatial_indices(B, N)
        point_indices = self._generate_point_indices(B, N)
        
        # Sample features for each camera
        all_sampled_features = []
        for camera_id, camera_features in features.items():
            # Sample temporal frames
            temporal_features = self._sample_temporal_frames(camera_features, temporal_indices)
            
            # Sample spatial frames
            spatial_features = self._sample_spatial_frames(temporal_features, spatial_indices)
            
            # Sample points
            sampled_features = self._sample_points(spatial_features, point_indices)
            
            all_sampled_features.append(sampled_features)
        
        # Concatenate features from all cameras
        sampled_features = torch.cat(all_sampled_features, dim=-1)
        
        return {
            'sampled_features': sampled_features,
            'temporal_indices': temporal_indices,
            'spatial_indices': spatial_indices,
            'point_indices': point_indices
        }
    
    def _generate_temporal_indices(self, B: int, N: int) -> torch.Tensor:
        """Generate indices for temporal frame sampling.
        
        Args:
            B: Batch size
            N: Number of anchors
            
        Returns:
            torch.Tensor: Indices with shape [B, N, num_temporal_frames]
        """
        indices = torch.randint(0, self.num_temporal_frames, (B, N, self.num_temporal_frames))
        return indices
    
    def _generate_spatial_indices(self, B: int, N: int) -> torch.Tensor:
        """Generate indices for spatial frame sampling.
        
        Args:
            B: Batch size
            N: Number of anchors
            
        Returns:
            torch.Tensor: Indices with shape [B, N, num_spatial_frames]
        """
        indices = torch.randint(0, self.num_spatial_frames, (B, N, self.num_spatial_frames))
        return indices
    
    def _generate_point_indices(self, B: int, N: int) -> torch.Tensor:
        """Generate indices for point sampling.
        
        Args:
            B: Batch size
            N: Number of anchors
            
        Returns:
            torch.Tensor: Indices with shape [B, N, num_points, 2]
        """
        indices = torch.randint(0, self.num_points, (B, N, self.num_points, 2))
        return indices
    
    @classmethod
    def build(cls, cfg: Dict) -> 'MultiviewTemporalSpatialFeatureSampler':
        """Build a feature sampler from config.
        
        Args:
            cfg: Configuration dictionary
            
        Returns:
            MultiviewTemporalSpatialFeatureSampler instance
        """
        return cls(**cfg) 