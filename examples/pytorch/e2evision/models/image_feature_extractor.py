import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional, List, Union, Tuple
import numpy as np
import torch.nn.functional as F
from timm.models import create_model
from base import SourceCameraId, TrajParamIndex, CameraParamIndex, EgoStateIndex, CameraType


class FPNFeatureFusion(nn.Module):
    """
    FPN feature fusion module that fuses features from different scales through a top-down path
    """
    def __init__(
        self,
        feature_scales: List[int],
        in_channels_dict: Dict[int, int],
        fpn_channels: int = 256,
        out_channels: int = 256,
        downsample_scale: int = 4,
    ):
        super().__init__()
        
        self.feature_scales = feature_scales
        self.downsample_scale = downsample_scale
        self.fpn_channels = fpn_channels
        # FPN Lateral (1x1 conv) layers to adjust channels for each scale
        self.lateral_convs = nn.ModuleDict()
        for scale in self.feature_scales:
            self.lateral_convs[str(scale)] = nn.Conv2d(in_channels_dict[scale], fpn_channels, kernel_size=1)
            
        # Global context module to enhance features with larger receptive field
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Conv2d(in_channels_dict[max(self.feature_scales)], fpn_channels, kernel_size=1),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final refinement conv for the output scale
        self.output_conv = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final 1x1 conv to adjust output channels if needed
        self.final_conv = nn.Conv2d(fpn_channels, out_channels, kernel_size=1)
    
    def _upsample_add(self, x, y):
        """Upsample x and add it to y."""
        _, _, h, w = y.size()
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False) + y
        
    def forward(self, features: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Fuse features from different scales
        
        Args:
            features: Dictionary mapping different scales (2,4,8,16,32) to feature tensors
            
        Returns:
            Fused feature tensor at the downsample_scale
        """
        # Apply lateral convolutions to standardize channel dimensions
        laterals = {}
        for scale in self.feature_scales:
            laterals[scale] = self.lateral_convs[str(scale)](features[scale])
        
        # Extract global context from the deepest feature
        max_scale = max(self.feature_scales)
        global_ctx = self.global_context(features[max_scale])
        
        # Top-down pathway (from deep to shallow layers)
        prev_features = laterals[max_scale] + global_ctx.expand_as(laterals[max_scale])
        
        # Process scales from deep to shallow (excluding the deepest)
        for scale in sorted(self.feature_scales, reverse=True)[1:]:
            prev_features = self._upsample_add(prev_features, laterals[scale])
            if scale == self.downsample_scale:
                # Refine features at the output scale
                output_feat = self.output_conv(prev_features)
                # Adjust channels for final output
                output_feat = self.final_conv(output_feat)
                return output_feat
        
        return self.final_conv(self.output_conv(prev_features))


class FPNImageFeatureExtractor(nn.Module):
    """
    Enhanced image feature extraction module with FPN structure and single channel output:
    1. Multi-image input support
    2. FPN architecture to enhance features with global context
    3. Single unified output channel for each camera
    """
    def __init__(
        self, 
        camera_ids: List[SourceCameraId],
        downsample_scale: int = 4,  # Single scale to output (2, 4, 8, 16, or 32)
        fpn_channels: int = 256,  # Internal FPN channel dimension
        out_channels: int = 256,  # Final output channel dimension
        use_pretrained: bool = True,
        backbone: str = 'resnet18',
    ):
        super().__init__()
        
        self.camera_ids = camera_ids
        self.downsample_scale = downsample_scale
        self.fpn_channels = fpn_channels
        self.out_channels = out_channels
        
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
        assert downsample_scale in self.model_info['scales'], f"downsample_scale must be one of {self.model_info['scales']}, got {downsample_scale}"
        
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

        # Create a dictionary mapping scales to their channel dimensions
        feature_scales = [s for s in self.model_info['scales'] if s >= downsample_scale]
        in_channels_dict = {scale: self.model_info['channels'][i] for i, scale in enumerate(self.model_info['scales'])}
        
        # Create the FPN fusion module
        self.fpn_fusion = FPNFeatureFusion(
            feature_scales=feature_scales,
            in_channels_dict=in_channels_dict,
            fpn_channels=fpn_channels,
            out_channels=out_channels,
            downsample_scale=downsample_scale
        )
        
        if not use_pretrained:
            self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, nonlinearity='relu')
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FPN architecture for enhanced features with single channel output.
        
        Args:
            x: Tensor of shape [B, C, H, W]
            
        Returns:
            Tensor of shape [B, C_out, H_out, W_out]
        """
        backbone_features = {}
        # Extract backbone features at different scales
        if self.model_info['type'] == 'torchvision':
            feat = x
            for i, layer in enumerate(self.features):
                feat = layer(feat)
                scale = self.model_info['scales'][i]
                backbone_features[scale] = feat
                
        else:  # Timm models
            all_feats = self.backbone(x)
            backbone_features = {scale: feat for scale, feat in zip(self.model_info['scales'], all_feats)}
        
        # for scale in backbone_features:
        #     print(f'backbone name {self.model_info["model"]} scale {scale} shape {backbone_features[scale].shape}')
            
        # Use the FPN fusion module to process features
        results = self.fpn_fusion(backbone_features)
        
        return results


class ImageFeatureExtractor(nn.Module):
    """Image feature extraction module with flexible backbone support."""
    def __init__(
        self,
        camera_ids: List[SourceCameraId],
        out_channels: int = 256,
        use_pretrained: bool = True,
        backbone: str = 'resnet18',
    ):
        super().__init__()
        
        self.camera_ids = camera_ids
        
        # 扩展支持的 backbone 列表（新增轻量化大感受野模型）
        self.supported_backbones = {
            # ==================== torchvision 模型 ====================
            'resnet18': {'model': models.resnet18, 'weights': models.ResNet18_Weights.DEFAULT, 'out_features': 512},
            'resnet50': {'model': models.resnet50, 'weights': models.ResNet50_Weights.DEFAULT, 'out_features': 2048},
            'mobilenet_v2': {'model': models.mobilenet_v2, 'weights': models.MobileNet_V2_Weights.DEFAULT, 'out_features': 1280},
            'mobilenet_v3_small': {'model': models.mobilenet_v3_small, 'weights': models.MobileNet_V3_Small_Weights.DEFAULT, 'out_features': 576},
            # ==================== timm 模型 ====================
            'efficientnet_b0': {'model': 'efficientnet_b0', 'out_features': 1280},
            'efficientnet_b1': {'model': 'efficientnet_b1', 'out_features': 1280},
            'repvgg_a0': {'model': 'repvgg_a0', 'out_features': 1280},
            'repvgg_a1': {'model': 'repvgg_a1', 'out_features': 1280},
            'convnext_tiny': {'model': 'convnext_tiny', 'out_features': 768},
            'mobilevitv2_050': {'model': 'mobilevitv2_050', 'out_features': 256},
            'efficientformerv2_s0': {'model': 'efficientformerv2_s0', 'out_features': 256},
            'mobileone_s0': {'model': 'mobileone_s0', 'out_features': 1024},
            'ghostnet_100': {'model': 'ghostnet_100', 'out_features': 160},
        }
        
        # 检查 backbone 是否支持
        if backbone not in self.supported_backbones:
            raise ValueError(f"Unsupported backbone: {backbone}. Supported: {list(self.supported_backbones.keys())}")
        
        # 加载 backbone
        model_info = self.supported_backbones[backbone]
        if 'weights' in model_info:  # torchvision 模型
            self.backbone = model_info['model'](weights=model_info['weights'] if use_pretrained else None)
            if 'resnet' in backbone:
                self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            elif 'mobilenet' in backbone:
                self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        else:  # timm 模型
            self.backbone = create_model(
                model_info['model'],
                pretrained=use_pretrained,
                features_only=True
            )
        
        # 获取输出通道数
        out_features = model_info['out_features']
        
         # 通道调整层（新增动态卷积核大小支持）
        self.channel_adjust = nn.Sequential(
            nn.Conv2d(out_features, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        if not use_pretrained:
            self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):  # 处理 timm 的多尺度输出
            x = x[-1]
        x = self.channel_adjust(x)
        return x


if __name__ == '__main__':
    import pdb; pdb.set_trace()
    model = FPNImageFeatureExtractor(camera_ids=[SourceCameraId.FRONT_LEFT])
    model.forward(torch.randn(1, 3, 640, 640))