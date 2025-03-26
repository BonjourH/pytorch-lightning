import torch
import torch.nn as nn
from xinnovation.src.core.registry import BACKBONES

@BACKBONES.register_module()
class ResNet(nn.Module):
    """ResNet backbone with configurable depth and features.
    
    Args:
        depth (int): Depth of ResNet (18, 34, 50, 101, 152)
        in_channels (int): Number of input channels
        pretrained (bool, str): Whether to use pretrained weights
        zero_init_residual (bool): Whether to zero-initialize the last BN layer
    """
    
    def __init__(
        self,
        depth: int = 50,
        in_channels: int = 3,
        pretrained: bool = False,
        zero_init_residual: bool = False
    ):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pretrained = pretrained
        
        # Define block configurations based on depth
        block_configs = {
            18: (2, 2, 2, 2),
            34: (3, 4, 6, 3),
            50: (3, 4, 6, 3),
            101: (3, 4, 23, 3),
            152: (3, 8, 36, 3)
        }
        
        if depth not in block_configs:
            raise ValueError(f"Unsupported ResNet depth: {depth}")
            
        self.block_config = block_configs[depth]
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=False)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, self.block_config[0])
        self.layer2 = self._make_layer(64, 128, self.block_config[1], stride=2)
        self.layer3 = self._make_layer(128, 256, self.block_config[2], stride=2)
        self.layer4 = self._make_layer(256, 512, self.block_config[3], stride=2)
        
        # Initialize weights
        self._initialize_weights(zero_init_residual)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # First block with stride
        layers.append(
            self._make_block(in_channels, out_channels, stride=stride)
        )
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(
                self._make_block(out_channels, out_channels)
            )
            
        return nn.Sequential(*layers)
    
    def _make_block(self, in_channels, out_channels, stride=1):
        if self.depth >= 50:
            return self._make_bottleneck(in_channels, out_channels, stride)
        else:
            return self._make_basic_block(in_channels, out_channels, stride)
    
    def _make_basic_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_bottleneck(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 4, 1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, nn.Sequential):
                    for block in m:
                        if isinstance(block, (self._make_basic_block, self._make_bottleneck)):
                            nn.init.constant_(block[-1].weight, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x 