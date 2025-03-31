import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional, List, Union, Tuple
import numpy as np
import torch.nn.functional as F
from timm.models import create_model
from xinnovation.src.core.registry import BACKBONES


@BACKBONES.register_module()
class ImageFeatureExtractor(nn.Module):
    """
    Image feature extraction module for multi-scale features:
    1. Multi-scale feature extraction from backbone
    2. Support for various pretrained backbones
    3. Returns features at different scales
    """
    def __init__(self, backbone: str, scales_to_drop: List[int] = [], use_pretrained: bool = True, **kwargs):
        super().__init__()
        """
        Args:
            backbone: str, backbone name
            scales_to_drop: List[int], if the scale is in the list, its corresponding feature will be dropped
            use_pretrained: bool, whether to use pretrained backbone
        """
        
        # Supported backbones dictionary with output channels at each stage
        self.supported_backbones = {
            # ==================== torchvision models ====================
            'resnet18': {
                'type': 'torchvision',
                'model': models.resnet18, 
                'weights': models.ResNet18_Weights.DEFAULT,
                'channels': [64, 128, 256, 512],
                'scales': [4, 8, 16, 32]
            },
            'resnet50': {
                'type': 'torchvision',
                'model': models.resnet50, 
                'weights': models.ResNet50_Weights.DEFAULT,
                'channels': [256, 512, 1024, 2048],
                'scales': [4, 8, 16, 32]
            },
            'mobilenet_v2': {
                'type': 'torchvision',
                'model': models.mobilenet_v2, 
                'weights': models.MobileNet_V2_Weights.DEFAULT,
                'channels': [24, 32, 96, 1280],
                'scales': [2, 4, 8, 16]         # 实际下采样倍数
            },
            'mobilenet_v3_small': {
                'type': 'torchvision',
                'model': models.mobilenet_v3_small, 
                'weights': models.MobileNet_V3_Small_Weights.DEFAULT,
                'channels': [16, 24, 48, 576],
                'scales': [2, 4, 8, 16]
            },
            # ==================== timm models ====================
            'efficientnet_b0': {
                'type': 'timm',
                'model': 'efficientnet_b0',
                'channels': [24, 40, 112, 1280],
                'scales': [2, 4, 8, 16]
            },
            'efficientnet_b1': {
                'type': 'timm',
                'model': 'efficientnet_b1',
                'channels': [24, 40, 112, 1280],
                'scales': [2, 4, 8, 16]
            },
            'repvgg_a0': {
                'type': 'timm',
                'model': 'repvgg_a0',
                'channels': [48, 48, 96, 192],
                'scales': [2, 4, 8, 16]
            },
            'repvgg_a1': {
                'type': 'timm',
                'model': 'repvgg_a1',
                'channels': [64, 64, 128, 256],
                'scales': [2, 4, 8, 16]
            },
            'repvgg_a2': {
                'type': 'timm',
                'model': 'repvgg_a2',
                'channels': [64, 96, 192, 384, 1408],
                'scales': [2, 4, 8, 16, 32]
            },
            'convnext_tiny': {
                'type': 'timm',
                'model': 'convnext_tiny', 
                'channels': [96, 192, 384, 768],
                'scales': [2, 4, 8, 16]
            },
            'mobilevitv2_050': {
                'type': 'timm',
                'model': 'mobilevitv2_050', 
                'channels': [64, 128, 192, 256],
                'scales': [2, 4, 8, 16]
            },
            'efficientformerv2_s0': {
                'type': 'timm',
                'model': 'efficientformerv2_s0',
                'channels': [32, 64, 128, 256],
                'scales': [2, 4, 8, 16]
            },
            'mobileone_s0': {
                'type': 'timm',
                'model': 'mobileone_s0', 
                'channels': [48, 96, 288, 1024],
                'scales': [2, 4, 8, 16]
            },
            'ghostnet_100': {
                'type': 'timm',
                'model': 'ghostnet_100', 
                'channels': [40, 80, 112, 160],
                'scales': [2, 4, 8, 16]
            },
        }
        
        # Check if backbone is supported
        if backbone not in self.supported_backbones:
            raise ValueError(f"Unsupported backbone: {backbone}. Supported: {list(self.supported_backbones.keys())}")
        
        # Load backbone
        self.model_info = self.supported_backbones[backbone]
        vals = [(i, j) for i, j in zip(self.model_info['scales'], self.model_info['channels']) if i not in scales_to_drop]
        self.scales = [i for i, _ in vals]
        self.out_channels = [j for _, j in vals]
        
        if self.model_info['type'] == 'torchvision':
            # Torchvision models
            self.backbone = self.model_info['model'](weights=self.model_info['weights'] if use_pretrained else None)
            
            # Prepare for feature extraction at multiple scales
            if 'resnet' in backbone:
                # For ResNet, we need to split the model to access intermediate features
                self.features = nn.ModuleList([
                    nn.Sequential(
                        self.backbone.conv1,
                        self.backbone.bn1,
                        self.backbone.relu,
                        self.backbone.maxpool,
                        self.backbone.layer1
                    ),  # 4× downsampling
                    self.backbone.layer2,  # 8× downsampling
                    self.backbone.layer3,  # 16× downsampling
                    self.backbone.layer4,  # 32× downsampling
                ])
            elif 'mobilenet' in backbone:
                # For MobileNet, we need to handle features differently
                features = list(self.backbone.features)
                # Split the features based on stride to get different scales
                self.features = nn.ModuleList([
                    nn.Sequential(*features[:4]),    # ~4× downsampling
                    nn.Sequential(*features[4:7]),   # ~8× downsampling
                    nn.Sequential(*features[7:14]),  # ~16× downsampling
                    nn.Sequential(*features[14:])    # ~32× downsampling
                ])
        else:
            # Timm models with features_only=True to get intermediate features
            self.backbone = create_model(
                self.model_info['model'],
                pretrained=use_pretrained,
                features_only=True
            )
            
        if not use_pretrained:
            self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, nonlinearity='relu')
            
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Forward pass to extract multi-scale features.
        
        Args:
            x: Tensor of shape [B, C, H, W]
            
        Returns:
            List[torch.Tensor]: List of features from different stages, from high to low resolution
            List[int]: List of scales, from high to low resolution
        """
        backbone_features = []
        scales = []
        
        # Extract backbone features at different scales
        if self.model_info['type'] == 'torchvision':
            feat = x
            for i, layer in enumerate(self.features):
                feat = layer(feat)
                scale = self.model_info['scales'][i]
                if scale in self.scales:
                    backbone_features.append(feat)
                    scales.append(scale)
        else:  # Timm models
            all_feats = self.backbone(x)
            for i, scale in enumerate(self.model_info['scales']):
                if scale in self.scales:
                    backbone_features.append(all_feats[i])
                    scales.append(scale)
        
        return backbone_features, scales
