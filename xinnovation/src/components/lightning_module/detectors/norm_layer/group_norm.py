import torch
import torch.nn as nn
from xinnovation.src.core.registry import NORM_LAYERS

@NORM_LAYERS.register_module()
class GroupNorm(nn.Module):
    """Group Normalization layer.
    
    Args:
        num_groups (int): Number of groups for group normalization
        num_channels (int): Number of channels in the input
        eps (float): A value added to the denominator for numerical stability
        affine (bool): Whether to use learnable affine parameters
    """
    
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True
    ):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
            
    def forward(self, x):
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Normalized tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Reshape for group normalization
        x = x.view(B, self.num_groups, -1)
        
        # Compute mean and variance
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        
        # Normalize
        x = (x - mean) / (torch.sqrt(var + self.eps))
        
        # Reshape back
        x = x.view(B, C, H, W)
        
        # Apply affine transformation
        if self.affine:
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
            
        return x 