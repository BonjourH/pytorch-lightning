import torch
import torch.nn as nn
from typing import Literal, Optional

class BaseTemporalFusion(nn.Module):
    """Base class for temporal feature fusion strategies."""
    def __init__(self):
        super().__init__()
        
    def _init_weights(self):
        """Initialize network weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for temporal fusion.
        
        Args:
            x (torch.Tensor): Input tensor with shape [B, T, C, H, W]
        
        Returns:
            torch.Tensor: Fused feature tensor
        """
        raise NotImplementedError("Subclasses must implement forward method")

class SimpleAverageFusion(BaseTemporalFusion):
    """Simple average fusion across temporal dimension."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute mean across temporal dimension.
        
        Args:
            x (torch.Tensor): Input features [B, T, C, H, W]
        
        Returns:
            torch.Tensor: Averaged features [B, C, H, W]
        """
        return x.mean(dim=1)

class SelfAttentionFusion(BaseTemporalFusion):
    """Self-attention based temporal fusion."""
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)
        self._init_weights()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention across temporal features.
        
        Args:
            x (torch.Tensor): Input features [B, T, C, H, W]
        
        Returns:
            torch.Tensor: Fused features [B, C, H, W]
        """
        B, T, C, H, W = x.shape
        
        # Flatten spatial dimensions for attention
        x_flat = x.reshape(B, T, C * H * W).contiguous()
        
        # Self-attention
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        
        # Residual connection and normalization
        x_flat = self.norm(x_flat + attn_out)
        
        # Reshape and average over time
        x = x_flat.reshape(B, T, C, H, W)
        return x.mean(dim=1)

class CrossAttentionFusion(BaseTemporalFusion):
    """Cross-attention based temporal fusion."""
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)
        self._init_weights()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention using first frame as query.
        
        Args:
            x (torch.Tensor): Input features [B, T, C, H, W]
        
        Returns:
            torch.Tensor: Fused features [B, C, H, W]
        """
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B, T, C * H * W)
        
        # Use first frame as query
        query = x_flat[:, 0:1]
        cross_attn_out, _ = self.cross_attention(query, x_flat, x_flat)
        
        # Residual connection and normalization
        cross_attn_out = self.norm(query + cross_attn_out)
        
        # Reshape and average
        x = cross_attn_out.reshape(B, 1, C, H, W)
        return x.mean(dim=1)

class GatedTemporalFusion(BaseTemporalFusion):
    """Gated temporal fusion mechanism."""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1)
        self._init_weights()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply gated fusion across temporal dimension.
        
        Args:
            x (torch.Tensor): Input features [B, T, C, H, W]
        
        Returns:
            torch.Tensor: Fused features [B, C, H, W]
        """
        x_perm = x.permute(0, 2, 1, 3, 4)
        gate = self.gate(x_perm)
        fused = self.fusion_conv(x_perm)
        
        # Gated weighting
        output = (fused * gate).mean(dim=2)
        return output

class RNNTemporalFusion(BaseTemporalFusion):
    """RNN-based temporal feature fusion."""
    def __init__(self, feature_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or feature_dim
        self.gru = nn.GRU(
            input_size=feature_dim * (feature_dim * feature_dim), 
            hidden_size=hidden_dim, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, feature_dim * feature_dim)
        self._init_weights()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process temporal features using GRU.
        
        Args:
            x (torch.Tensor): Input features [B, T, C, H, W]
        
        Returns:
            torch.Tensor: Fused features [B, C, H, W]
        """
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B, T, -1)
        
        # GRU processing
        _, hidden = self.gru(x_flat)
        
        # Reshape to original feature dimensions
        output = self.fc(hidden.squeeze(0)).reshape(B, C, H, W)
        return output

class TemporalFusionFactory:
    """Factory for creating temporal fusion strategies."""
    @staticmethod
    def create(
        strategy: Literal['average', 'self_attention', 'cross_attention', 'gated', 'rnn'] = 'average', 
        **kwargs
    ) -> BaseTemporalFusion:
        """
        Create a temporal fusion strategy.
        
        Args:
            strategy (str): Type of fusion strategy
            **kwargs: Initialization parameters for the strategy
        
        Returns:
            BaseTemporalFusion: Instantiated fusion module
        
        Raises:
            ValueError: If an unsupported strategy is specified
        """
        strategies = {
            'average': SimpleAverageFusion,
            'self_attention': SelfAttentionFusion,
            'cross_attention': CrossAttentionFusion,
            'gated': GatedTemporalFusion,
            'rnn': RNNTemporalFusion
        }
        
        if strategy not in strategies:
            raise ValueError(f"Unsupported fusion strategy: {strategy}")
        
        return strategies[strategy](**kwargs)
    