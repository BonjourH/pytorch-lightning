import torch
import torch.nn as nn
from xinnovation.src.core.registry import HEADS

@HEADS.register_module()
class DetectionHead(nn.Module):
    """Basic detection head with classification and regression branches.
    
    Args:
        in_channels (int): Number of input channels
        num_classes (int): Number of object classes
        num_anchors (int): Number of anchors per location
        feat_channels (int): Number of channels in intermediate features
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 9,
        feat_channels: int = 256
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Shared feature layers
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True)
        )
        
        # Classification branch
        self.cls_conv = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True)
        )
        self.cls_out = nn.Conv2d(
            feat_channels,
            num_classes * num_anchors,
            1
        )
        
        # Regression branch
        self.reg_conv = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True)
        )
        self.reg_out = nn.Conv2d(
            feat_channels,
            4 * num_anchors,
            1
        )
        
    def forward(self, x):
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input feature map
            
        Returns:
            tuple: Classification and regression predictions
        """
        # Shared features
        shared_feat = self.shared_conv(x)
        
        # Classification branch
        cls_feat = self.cls_conv(shared_feat)
        cls_pred = self.cls_out(cls_feat)
        
        # Regression branch
        reg_feat = self.reg_conv(shared_feat)
        reg_pred = self.reg_out(reg_feat)
        
        return cls_pred, reg_pred 