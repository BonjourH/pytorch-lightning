import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np

from base import SourceCameraId, TrajParamIndex, ObjectType, CameraParamIndex, EgoStateIndex
from .components import TrajectoryDecoder
from .temporal_fusion_layer import TemporalFusionFactory
from .image_feature_extractor import ImageFeatureExtractor

    
class E2EPerceptionNet(nn.Module):
    """End-to-end perception network."""
    
    def __init__(self,
                 camera_ids: List[SourceCameraId],
                 feature_dim: int = 256,
                 num_queries: int = 128,
                 num_decoder_layers: int = 6,
                 use_pretrained: bool = False,
                 backbone: str = 'resnet50'):
        super().__init__()
        self.camera_ids = camera_ids
        
        # Create feature extractors for each camera
        self.feature_extractors = nn.ModuleDict({
            str(camera_id.value): ImageFeatureExtractor([camera_id], feature_dim, use_pretrained=use_pretrained, backbone=backbone)
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
        all_features_dict = {}
        for camera_id in self.camera_ids:
            images = batch['images'][camera_id]  # [B, T, C, H, W]
            B, T, C, H, W = images.shape
            
            images = images.view(B * T, C, H, W)
            feat = self.feature_extractors[str(camera_id.value)](images)
            _, C0, H0, W0 = feat.shape
            feat = feat.view(B, T, C0, H0, W0)
            all_features_dict[camera_id] = feat
        
        # Fuse temporal features for each camera
        fused_features_dict = {}
        for camera_id, features in all_features_dict.items():
            fused_features_dict[camera_id] = self.temporal_fusion(features)
        
        # Decode trajectories
        outputs = self.decoder(fused_features_dict, batch['calibrations'], batch['ego_states'])
        
        return outputs 