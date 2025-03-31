import torch
import torch.nn as nn
from xinnovation.src.core.registry import NECKS
from typing import List

@NECKS.register_module()
class FPN(nn.Module):
    """Feature Pyramid Network.
    
    Args:
        in_channels (list): Number of channels for each input feature map
        out_channels (int): Number of channels for each output feature map
        extra_blocks (int): Number of extra top-down blocks
        relu_before_extra_convs (bool): Whether to use ReLU before extra convs
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        extra_blocks: int = 0,
        relu_before_extra_convs: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.extra_blocks = extra_blocks
        self.relu_before_extra_convs = relu_before_extra_convs
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList()
        for in_channel in in_channels:
            self.lateral_convs.append(
                nn.Conv2d(in_channel, out_channels, 1)
            )
            
        # Smooth layers
        self.smooth_convs = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.smooth_convs.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )
            
        # Extra top-down blocks
        self.extra_blocks = extra_blocks
        if extra_blocks > 0:
            self.extra_convs = nn.ModuleList()
            for _ in range(extra_blocks):
                self.extra_convs.append(
                    nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
                )
                
    def forward(self, inputs):
        """Forward pass.
        
        Args:
            inputs (list): List[torch.Tensor] features from different stages, from high to low resolution
            
        Returns:
            list: List[torch.Tensor]
        """
        assert len(inputs) == len(self.in_channels)
        
        # Build lateral connections
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[-2:] # (H, W)
            laterals[i - 1] += nn.functional.interpolate(
                laterals[i], size=prev_shape, mode='bilinear'
            )
            
        # Smooth lateral connections
        for i in range(used_backbone_levels - 1):
            laterals[i] = self.smooth_convs[i](laterals[i])
            
        # Add extra top-down blocks
        if self.extra_blocks > 0:
            last_feat = laterals[-1]
            for i in range(self.extra_blocks):
                if self.relu_before_extra_convs:
                    last_feat = nn.functional.relu(last_feat)
                last_feat = self.extra_convs[i](last_feat)
                laterals.append(last_feat)
                
        return laterals 