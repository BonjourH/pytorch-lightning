import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np

from base import SourceCameraId, TrajParamIndex, ObjectType, CameraParamIndex, EgoStateIndex
from .components import TrajectoryDecoder
from .temporal_fusion_layer import TemporalFusionFactory
from .image_feature_extractor import FPNImageFeatureExtractor
from configs.config import ModelConfig, DataConfig

    
class E2EPerceptionNet(nn.Module):
    """End-to-end perception network."""
    
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__()
        
        # Create feature extractors for each camera group
        self.feature_extractors = nn.ModuleDict({
            cfg.name: FPNImageFeatureExtractor(
                camera_ids=cfg.camera_group,
                downsample_scale=cfg.downsample_scale,
                fpn_channels=cfg.fpn_channels,
                out_channels=cfg.out_channels,
                use_pretrained=cfg.use_pretrained,
                backbone=cfg.backbone
            )
            for cfg in data_config.camera_groups
        })
        
        # Temporal fusion for each camera
        self.temporal_fusion = TemporalFusionFactory.create(strategy='average')
        
        # Trajectory decoder
        self.decoder = TrajectoryDecoder(model_config.decoder.num_layers,
                                        model_config.decoder.num_queries,
                                        model_config.decoder.feature_dim,
                                        model_config.decoder.hidden_dim,
                                        model_config.decoder.num_points)
    
    def forward(self, batch: Dict) -> List[Dict[str, torch.Tensor]]:
        """Forward pass.
        
        Args:
            batch: Dictionary containing:
                - images: Dict[str -> Tensor[B, T, N_cams, C, H, W]]
                - calibrations: Dict[camera_id -> Tensor[B, CameraParamIndex.END_OF_INDEX]]
                - ego_states: Tensor[B, T, EgoStateIndex.END_OF_INDEX]
            
        Returns:
            List of decoder outputs at each layer
        """
        
        # Extract features from each camera
        all_features_dict = {}
        for name, imgs in batch['images'].items():
            B, T, N_cams, C, H, W = imgs.shape
            imgs = imgs.view(B * T * N_cams, C, H, W)
            feat = self.feature_extractors[name](imgs)
            down_scale = self.feature_extractors[name].downsample_scale
            output_channels = self.feature_extractors[name].out_channels
            feat = feat.view(B, T, N_cams, output_channels, H // down_scale, W // down_scale)
            for idx, camera_id in enumerate(self.feature_extractors[name].camera_ids):
                all_features_dict[camera_id] = feat[:, :, idx]
        
        # Fuse temporal features for each camera
        fused_features_dict = {}
        for camera_id, features in all_features_dict.items():
            fused_features_dict[camera_id] = self.temporal_fusion(features)
        
        # Decode trajectories
        outputs = self.decoder(fused_features_dict, batch['calibrations'], batch['ego_states'])
        
        return outputs 