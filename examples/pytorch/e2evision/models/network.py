import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np

from base import SourceCameraId, TrajParamIndex, ObjectType, AttributeType, CameraParamIndex, EgoStateIndex
from .components import ImageFeatureExtractor, TrajectoryDecoderLayer, TrajectoryDecoder
from .temporal_fusion_layer import TemporalFusionFactory

    
class E2EPerceptionNet(nn.Module):
    """End-to-end perception network."""
    
    def __init__(self,
                 camera_ids: List[SourceCameraId],
                 feature_dim: int = 256,
                 num_queries: int = 100,
                 num_decoder_layers: int = 6,
                 use_pretrained: bool = False,
                 backbone: str = 'resnet50'):
        super().__init__()
        self.camera_ids = camera_ids
        
        # Create feature extractors for each camera
        self.feature_extractors = nn.ModuleDict({
            str(camera_id.value): ImageFeatureExtractor(feature_dim, use_pretrained=use_pretrained, backbone=backbone)
            for camera_id in camera_ids
        })
        
        # Temporal fusion for each camera
        self.temporal_fusion = TemporalFusionFactory.create(strategy='average')
        
        # Trajectory decoder
        self.decoder = TrajectoryDecoder(
            num_layers=num_decoder_layers,
            num_queries=num_queries,
            feature_dim=feature_dim
        )
        
        # Initialize parameters
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, batch: Dict) -> List[Dict[str, torch.Tensor]]:
        """Forward pass.
        
        Args:
            batch: Dictionary containing:
                - images: Dict[camera_id -> Tensor[B, T, C, H, W]]
                - calibrations: Dict[camera_id -> Tensor[B, CameraParamIndex.END_OF_INDEX]]
                - ego_states: Tensor[B, T, EgoStateIndex.END_OF_INDEX]
            
        Returns:
            List of decoder outputs at each layer
        """
        B = next(iter(batch['images'].values())).shape[0]
        
        # Extract features from each camera
        all_features = {}
        for camera_id in self.camera_ids:
            images = batch['images'][camera_id]  # [B, T, C, H, W]
            B, T, C, H, W = images.shape
            
            # Process each timestep
            features = []
            for t in range(T):
                feat = self.feature_extractors[str(camera_id.value)](images[:, t])
                features.append(feat)
            
            # Stack temporal features
            features = torch.stack(features, dim=1)  # [B, T, C, H, W]
            all_features[camera_id] = features
        
        # Fuse temporal features for each camera
        fused_features = []
        for camera_id, features in all_features.items():
            fused = self.temporal_fusion(features)
            fused_features.append(fused)
        
        # Average features across cameras
        features = torch.stack(fused_features).mean(0)  # [B, C, H, W]
        
        # Decode trajectories
        outputs = self.decoder(features)
        
        return outputs 