import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from xinnovation.src.core.registry import DETECTORS

FPN_IMAGE_FEATURE_EXTRACTOR = DETECTORS.create_child("fpn_image_feature_extractor")

__all__ = ["FPNImageFeatureExtractor"]

@FPN_IMAGE_FEATURE_EXTRACTOR.register_module()
class FPNImageFeatureExtractor(nn.Module):
    """Feature Pyramid Network (FPN) feature extractor for images.
    
    Extracts multi-scale features from input images using a backbone network
    and a Feature Pyramid Network. The backbone network can be ResNet or other
    convolutional networks.
    """
    
    def __init__(self,
                 in_channels: int = 3,
                 backbone_type: str = 'resnet50',
                 fpn_channels: int = 256,
                 use_checkpoint: bool = False,
                 freeze_backbone: bool = False,
                 **kwargs):
        """Initialize the FPN image feature extractor.
        
        Args:
            in_channels: Number of input channels (3 for RGB images)
            backbone_type: Type of backbone network ('resnet18', 'resnet34', 'resnet50', etc.)
            fpn_channels: Number of channels in the FPN layers
            use_checkpoint: Whether to use gradient checkpointing to save memory
            freeze_backbone: Whether to freeze the backbone network
        """
        super().__init__()
        self.in_channels = in_channels
        self.fpn_channels = fpn_channels
        self.backbone_type = backbone_type
        
        # Create backbone network
        self.backbone = self._create_backbone(backbone_type, in_channels)
        
        # Get the output channels of each stage in the backbone
        if 'resnet' in backbone_type:
            if backbone_type in ['resnet18', 'resnet34']:
                backbone_out_channels = [64, 128, 256, 512]
            else:  # resnet50, resnet101, resnet152
                backbone_out_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        # Create lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, fpn_channels, kernel_size=1) 
            for c in backbone_out_channels
        ])
        
        # Create output convolutions for each FPN level
        self.output_convs = nn.ModuleList([
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
            for _ in range(len(backbone_out_channels))
        ])
        
        # Extra FPN levels (optional: P6, P7 for RetinaNet, etc.)
        self.extra_convs = nn.ModuleList([
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=2, padding=1)
        ])
        
        # Initialize weights
        self._init_weights()
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Use gradient checkpointing if specified
        self.use_checkpoint = use_checkpoint
    
    def _create_backbone(self, backbone_type: str, in_channels: int) -> nn.Module:
        """Create backbone network.
        
        Args:
            backbone_type: Type of backbone network
            in_channels: Number of input channels
            
        Returns:
            nn.Module: Backbone network
        """
        if 'resnet' in backbone_type:
            import torchvision.models as models
            
            # Get the ResNet model
            if backbone_type == 'resnet18':
                backbone = models.resnet18(pretrained=True)
            elif backbone_type == 'resnet34':
                backbone = models.resnet34(pretrained=True)
            elif backbone_type == 'resnet50':
                backbone = models.resnet50(pretrained=True)
            elif backbone_type == 'resnet101':
                backbone = models.resnet101(pretrained=True)
            elif backbone_type == 'resnet152':
                backbone = models.resnet152(pretrained=True)
            else:
                raise ValueError(f"Unsupported ResNet type: {backbone_type}")
            
            # Modify the first conv layer if in_channels is not 3
            if in_channels != 3:
                backbone.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
            
            # Split the ResNet into stages
            stages = nn.ModuleList([
                nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool),
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4
            ])
            
            return stages
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
    
    def _init_weights(self):
        """Initialize the weights of the FPN layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass to extract features from input images.
        
        Args:
            x: Input images of shape [B, C, H, W]
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of multi-scale features
                {
                    'p3': features of shape [B, fpn_channels, H/8, W/8],
                    'p4': features of shape [B, fpn_channels, H/16, W/16],
                    'p5': features of shape [B, fpn_channels, H/32, W/32],
                    'p6': features of shape [B, fpn_channels, H/64, W/64],
                    'p7': features of shape [B, fpn_channels, H/128, W/128]
                }
        """
        # Extract features from the backbone
        features = []
        for i, stage in enumerate(self.backbone):
            if i == 0:
                x = stage(x)
            else:
                if self.use_checkpoint and self.training:
                    from torch.utils.checkpoint import checkpoint
                    x = checkpoint(stage, x)
                else:
                    x = stage(x)
                features.append(x)
        
        # Build FPN features (top-down pathway)
        lateral_features = [conv(feature) for feature, conv in zip(features, self.lateral_convs)]
        
        # Top-down pathway
        fpn_features = [lateral_features[-1]]  # Start with the topmost feature
        for i in range(len(lateral_features) - 2, -1, -1):
            # Upsample the higher-level feature
            top_down_feature = F.interpolate(
                fpn_features[-1], size=lateral_features[i].shape[2:], mode='nearest'
            )
            # Add the lateral connection
            fpn_features.append(top_down_feature + lateral_features[i])
        
        # Reverse the list to get features from low to high levels
        fpn_features = fpn_features[::-1]
        
        # Apply 3x3 conv to each merged feature
        fpn_features = [conv(feature) for feature, conv in zip(fpn_features, self.output_convs)]
        
        # Add extra FPN levels (P6, P7)
        p6 = self.extra_convs[0](fpn_features[-1])
        p7 = self.extra_convs[1](F.relu(p6))
        
        # Create output dictionary
        output_features = {
            'p3': fpn_features[0],  # 1/8 of input size
            'p4': fpn_features[1],  # 1/16 of input size
            'p5': fpn_features[2],  # 1/32 of input size
            'p6': p6,               # 1/64 of input size
            'p7': p7                # 1/128 of input size
        }
        
        return output_features
    
    @classmethod
    def build(cls, cfg: Dict) -> 'FPNImageFeatureExtractor':
        """Build a feature extractor from config.
        
        Args:
            cfg: Configuration dictionary
            
        Returns:
            FPNImageFeatureExtractor instance
        """
        return cls(**cfg) 