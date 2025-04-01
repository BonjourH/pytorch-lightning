import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Any, List
from xinnovation.src.core import SourceCameraId, DETECTORS, ANCHOR_GENERATOR, IMAGE_FEATURE_EXTRACTOR, PLUGINS
from xinnovation.src.components.lightning_module.detectors import FPNImageFeatureExtractor, Anchor3DGenerator
from .mts_feature_aggregator import MultiviewTemporalSpatialFeatureAggregator

__all__ = ["Sparse4DDetector"]

@DETECTORS.register_module()
class Sparse4DDetector(nn.Module):
    def __init__(self, anchor_generator: Dict,
                    camera_groups: Dict[str, List[SourceCameraId]],
                    feature_extractors: Dict, 
                    mts_feature_aggregator: Dict,
                    **kwargs):
        super().__init__()
        self.camera_groups = camera_groups
        self.anchor_generator = ANCHOR_GENERATOR.build(anchor_generator)

        # Create feature extractors for each camera group
        self.feature_extractors = nn.ModuleDict({
            name: IMAGE_FEATURE_EXTRACTOR.build(cfg)
            for name, cfg in feature_extractors.items()
        })
        
        self.mts_feature_aggregator = PLUGINS.build(mts_feature_aggregator)
        
    
    def _extract_features(self, group_name: str, imgs: torch.Tensor) -> List[torch.Tensor]:
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
        extractor = self.feature_extractors[group_name]
        feats = extractor(imgs)
        feats = [feat.view(B, T, N_cams, extractor.out_channel, H // ds, W // ds) for feat, ds in zip(feats, extractor.fpn_downsample_scales)]
        return feats
    
    def forward(self, batch: Dict) -> List[Dict[str, torch.Tensor]]:
        """Forward pass.
        
        Args:
            batch: Dictionary containing:
                - images: Dict[str -> Tensor[B, T, N_cams, C, H, W]], group_name -> img tensor
                - calibrations: Dict[camera_id -> Tensor[B, CameraParamIndex.END_OF_INDEX]]
                - ego_states: Tensor[B, T, EgoStateIndex.END_OF_INDEX]
                - camera_ids: List[SourceCameraId], camera ids
            
        Returns:
            List[Dict[str, torch.Tensor]], decoder outputs at each layer
        """
        
        # Extract features from each camera
        all_features_dict: Dict[SourceCameraId, List[torch.Tensor]] = {}  
        for group_name, imgs in batch['images'].items():
            B, T, N_cams = imgs.shape[:3]
            if self.use_checkpoint:
                feats = torch.utils.checkpoint.checkpoint(self._extract_features, group_name, imgs, use_reentrant=False)
            else:
                feats = self._extract_features(group_name, imgs) 
                # feats: List[Tensor[B*T*N_cams, C, H // down_scale_i, W // down_scale_i]]
            for i in range(len(feats)):
                C_i, H_i, W_i = feats[i].shape[2:]
                feats[i] = feats[i].view(B * T, N_cams, C_i, H_i, W_i)
            for idx, camera_id in enumerate(self.camera_groups[group_name]):
                all_features_dict[camera_id] = [feat[:, :, idx] for feat in feats] # (B * T, C, H // down_scale_i, W // down_scale_i)
                
        # Decoder part
        
        # Decode trajectories
        fused_features_dict = {}
        outputs = self.decoder(fused_features_dict, batch['calibrations'], batch['ego_states'])
        
        # Clear intermediate tensors to save memory
        # del all_features_dict
        # del fused_features_dict
        # torch.cuda.empty_cache()
        
        return outputs 