import torch
import torch.nn as nn
from xinnovation.src.core.registry import ATTENTION

@ATTENTION.register_module()
class SelfAttention(nn.Module):
    """Self-Attention module.
    
    Args:
        dim (int): Input feature dimension
        num_heads (int): Number of attention heads
        qkv_bias (bool): Whether to use bias in qkv projection
        attn_drop (float): Dropout rate for attention weights
        proj_drop (float): Dropout rate for output projection
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Attention dropout
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape
        
        # QKV projection and reshape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x 