import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from xinnovation.src.core.registry import COMPONENTS

__all__ = ["MultiviewTemporalSpatialFeatureAggregator"]

@COMPONENTS.register_module()
class MultiviewTemporalSpatialFeatureAggregator(nn.Module):
    """Aggregate features from multiple views and temporal frames.
    
    This class aggregates features sampled from multiple camera views and temporal frames
    for each anchor point in 3D space.
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 **kwargs):
        """Initialize the feature aggregator.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, out_channels)
        
        # Multi-head self-attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_channels, out_channels * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_channels * 4, out_channels)
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(out_channels)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(out_channels, out_channels)
        
    def _apply_attention(self,
                        x: torch.Tensor,
                        attention_layer: nn.MultiheadAttention,
                        ffn: nn.Sequential,
                        norm: nn.LayerNorm) -> torch.Tensor:
        """Apply attention and feed-forward network.
        
        Args:
            x: Input tensor with shape [B, N, C]
            attention_layer: Multi-head attention layer
            ffn: Feed-forward network
            norm: Layer normalization
            
        Returns:
            torch.Tensor: Output tensor with shape [B, N, C]
        """
        # Apply attention
        attn_output, _ = attention_layer(x, x, x)
        x = x + attn_output
        
        # Apply layer normalization
        x = norm(x)
        
        # Apply feed-forward network
        ffn_output = ffn(x)
        x = x + ffn_output
        
        return x
    
    def forward(self,
                sampled_features: torch.Tensor,
                temporal_indices: torch.Tensor,
                spatial_indices: torch.Tensor,
                point_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            sampled_features: Sampled features with shape [B, N, num_points, C]
            temporal_indices: Indices of temporal frames sampled [B, N, num_temporal_frames]
            spatial_indices: Indices of spatial frames sampled [B, N, num_spatial_frames]
            point_indices: Indices of points sampled [B, N, num_points, 2]
            
        Returns:
            torch.Tensor: Aggregated features with shape [B, N, C]
        """
        B, N, P, C = sampled_features.shape
        
        # Project input features
        x = self.input_proj(sampled_features)  # [B, N, P, C]
        
        # Reshape for transformer
        x = x.reshape(B * N, P, C)
        
        # Apply transformer layers
        for i in range(self.num_layers):
            x = self._apply_attention(
                x,
                self.attention_layers[i],
                self.ffn_layers[i],
                self.norm_layers[i]
            )
        
        # Project output features
        x = self.output_proj(x)  # [B * N, P, C]
        
        # Reshape back
        x = x.reshape(B, N, P, C)
        
        # Average pooling over points
        x = x.mean(dim=2)  # [B, N, C]
        
        return x
    
    @classmethod
    def build(cls, cfg: Dict) -> 'MultiviewTemporalSpatialFeatureAggregator':
        """Build a feature aggregator from config.
        
        Args:
            cfg: Configuration dictionary
            
        Returns:
            MultiviewTemporalSpatialFeatureAggregator instance
        """
        return cls(**cfg) 