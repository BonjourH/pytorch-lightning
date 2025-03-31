import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Any, List
from xinnovation.src.core.registry import DETECTORS
from easydict import EasyDict as edict
from .anchor_generator import Anchor3DGenerator
from .fpn_image_feature_extractor import FPNImageFeatureExtractor
from .mts_feature_sampler import MultiviewTemporalSpatialFeatureSampler
from .mts_feature_aggregator import MultiviewTemporalSpatialFeatureAggregator

__all__ = ["Sparse4DDetector"]

@DETECTORS.register_module()
class Sparse4DDetector(nn.Module):
    def __init__(self, anchor_generator: Dict, 
                    feature_extractors: Dict, 
                    mts_feature_sampler: Dict,
                    mts_feature_aggregator: Dict,
                    **kwargs):
        super().__init__()
        self.anchor_generator = Anchor3DGenerator.build(anchor_generator)

        # Create feature extractors for each camera group
        self.feature_extractors = nn.ModuleDict({
            cfg.name: FPNImageFeatureExtractor.build(cfg)
            for cfg in feature_extractors
        })

        self.mts_feature_sampler = MultiviewTemporalSpatialFeatureSampler.build(mts_feature_sampler)
        self.mts_feature_aggregator = MultiviewTemporalSpatialFeatureAggregator.build(mts_feature_aggregator)
        
    
    def _extract_features(self, name: str, imgs: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from images using the specified feature extractor.
        
        Args:
            name: str, name of the feature extractor
            imgs: torch.Tensor, images with shape [B, T, N_cams, C, H, W]
            
        Returns:
            List[torch.Tensor], features with shape [B, T, N_cams, out_channels, H // down_scale_i, W // down_scale_i] for i in range(len(down_scales))
        """
        B, T, N_cams, C, H, W = imgs.shape
        imgs = imgs.view(B * T * N_cams, C, H, W)
        feats = self.feature_extractors[name](imgs)
        output_channel = self.feature_extractors[name].out_channel
        down_scales = self.feature_extractors[name].fpn_downsample_scales   
        feats = [feat.view(B, T, N_cams, output_channel, H // down_scale, W // down_scale) for feat, down_scale in zip(feats, down_scales)]
        return feats
    
    def forward(self, batch: Dict) -> List[Dict[str, torch.Tensor]]:
        """Forward pass.
        
        Args:
            batch: Dictionary containing:
                - images: Dict[str -> Tensor[B, T, N_cams, C, H, W]], group_name -> img tensor
                - calibrations: Dict[camera_id -> Tensor[B, CameraParamIndex.END_OF_INDEX]]
                - ego_states: Tensor[B, T, EgoStateIndex.END_OF_INDEX]
            
        Returns:
            List[Dict[str, torch.Tensor]], decoder outputs at each layer
        """
        
        # Extract features from each camera with gradient checkpointing
        all_features_dict = {}
        for name, imgs in batch['images'].items():
            if self.use_checkpoint:
                feats = torch.utils.checkpoint.checkpoint(self._extract_features, name, imgs, use_reentrant=False)
            else:
                feats = self._extract_features(name, imgs)
                
            for idx, camera_id in enumerate(self.feature_extractors[name].camera_ids):
                all_features_dict[camera_id] = feats[:, :, idx]
        
        # Fuse temporal features for each camera
        # fused_features_dict = {}
        # for camera_id, features in all_features_dict.items():
        #     fused_features_dict[camera_id] = self.temporal_fusion(features)
        
        # Decode trajectories
        fused_features_dict = {}
        outputs = self.decoder(fused_features_dict, batch['calibrations'], batch['ego_states'])
        
        # Clear intermediate tensors to save memory
        # del all_features_dict
        # del fused_features_dict
        # torch.cuda.empty_cache()
        
        return outputs 