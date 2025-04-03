import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from xinnovation.src.core.registry import DETECTORS, BACKBONES, NECKS, IMAGE_FEATURE_EXTRACTOR
from .neck import FPN
from .backbone import ImageFeatureExtractor

# Create registry children


__all__ = ["FPNImageFeatureExtractor"]

@IMAGE_FEATURE_EXTRACTOR.register_module()
class FPNImageFeatureExtractor(nn.Module):
    """Feature Pyramid Network (FPN) feature extractor for images.
    
    Extracts multi-scale features from input images using a backbone network
    and a Feature Pyramid Network.
    """

    def __init__(self, backbone: Dict, neck: Dict, name: str="unknown", **kwargs):
        """Initialize the FPN image feature extractor.
        
        Args:
            backbone: Configuration for the backbone
            neck: Configuration for the neck
        """
        super().__init__()
        self.name = name
        
        # Create backbone
        self.backbone = BACKBONES.build(backbone)

        # Create neck with in_channels from backbone
        neck_config = neck.copy()
        neck_config['in_channels'] = self.backbone.out_channels
        self.neck = NECKS.build(neck_config)

    def out_channels(self):
        return self.neck.out_channels
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass to extract features from input images.
        
        Args:
            x: Input images of shape [B, C, H, W]

        Returns:
            List[torch.Tensor]: List of features from different stages, from high to low resolution
            List[int]: List of scales, from high to low resolution
        """
        # Extract features from the backbone
        features, scales = self.backbone(x)
        # Build FPN features
        output_features = self.neck(features)
        return output_features, scales
    