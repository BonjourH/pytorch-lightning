import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional

from base import SourceCameraId, ObjectType, TrajParamIndex

class ImageFeatureExtractor(nn.Module):
    """Image feature extraction module."""
    def __init__(self, out_channels: int = 256):
        super().__init__()
        
        # Use ResNet50 as backbone with updated weights parameter
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Remove classification head
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Add 1x1 conv to adjust channel dimension
        self.channel_adjust = nn.Conv2d(2048, out_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Features [B, out_channels, H/32, W/32]
        """
        x = self.backbone(x)
        x = self.channel_adjust(x)
        return x

class TemporalAttentionFusion(nn.Module):
    """Temporal attention fusion module."""
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, T, C, H, W]
        Returns:
            Fused features [B, C, H, W]
        """
        B, T, C, H, W = x.shape
        
        # Reshape for attention
        x = x.reshape(B, T, C * H * W).contiguous()
        
        # Self attention
        attn_out, _ = self.attention(x, x, x)
        
        # Add & norm
        x = self.norm(x + attn_out)
        
        # Reshape and mean over time
        x = x.reshape(B, T, C, H, W)
        x = x.mean(dim=1)
        
        return x

class TrajectoryRefinementLayer(nn.Module):
    """Layer for refining trajectory queries."""
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Feature attention
        self.feature_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
    def forward(self, queries: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: Tensor[B, N, hidden_dim]
            features: Tensor[B, N, hidden_dim]
            
        Returns:
            Tensor[B, N, hidden_dim]
        """
        # Self-attention
        q = self.norm1(queries)
        q = q + self.self_attn(q, q, q)[0]
        
        # Feature attention
        q = self.norm2(q)
        q = q + self.feature_attn(q, features, features)[0]
        
        # FFN
        q = self.norm3(q)
        q = q + self.ffn(q)
        
        return q

class TrajectoryDecoderLayer(nn.Module):
    """Single layer of trajectory decoder."""
    def __init__(self, feature_dim: int, num_heads: int = 8, dim_feedforward: int = 2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        
        self.linear1 = nn.Linear(feature_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, feature_dim)
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()
        
    def forward(self, 
                queries: torch.Tensor,
                memory: torch.Tensor,
                query_pos: Optional[torch.Tensor] = None,
                memory_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            queries: Object queries [B, num_queries, C]
            memory: Image features [B, H*W, C]
            query_pos: Query position encoding
            memory_pos: Memory position encoding
        Returns:
            Updated queries [B, num_queries, C]
        """
        # Self attention
        q = queries + query_pos if query_pos is not None else queries
        k = q
        v = queries
        queries2 = self.self_attn(q, k, v)[0]
        queries = self.norm1(queries + self.dropout(queries2))
        
        # Cross attention
        q = queries + query_pos if query_pos is not None else queries
        k = memory + memory_pos if memory_pos is not None else memory
        v = memory
        queries2 = self.cross_attn(q, k, v)[0]
        queries = self.norm2(queries + self.dropout(queries2))
        
        # Feed forward
        queries2 = self.linear2(self.dropout(self.activation(self.linear1(queries))))
        queries = self.norm3(queries + self.dropout(queries2))
        
        return queries 