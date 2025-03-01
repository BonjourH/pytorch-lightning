from typing import Dict, List, Optional
import torch
import torch.nn as nn
from base import SourceCameraId


class TemporalAttentionFusion(nn.Module):
    """Fuse temporal features using attention mechanism.
    Keep all K frames and fuse them using attention.
    """
    def __init__(self, 
                 feature_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Temporal attention for sequence fusion
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Position encodings for temporal attention
        self.temporal_pos_enc = nn.Parameter(torch.randn(1, 1, feature_dim))
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor[B, T, C, H, W]
                B: batch size
                T: sequence length
                C: number of channels
                H, W: height and width
        Returns:
            Tensor[B, T, C, H, W]: Temporally fused features
        """
        B, T, C, H, W = features.shape
        
        # Reshape for temporal attention: [B*H*W, T, C]
        feat = features.permute(0, 3, 4, 1, 2).reshape(-1, T, C)
        
        # Add temporal position encoding
        temp_pos = self.temporal_pos_enc.expand(feat.size(0), T, -1)
        feat = feat + temp_pos
        
        # Apply temporal attention
        fused_feat, _ = self.temporal_attention(feat, feat, feat)
        
        # Reshape back: [B, T, C, H, W]
        fused_feat = fused_feat.view(B, H, W, T, C).permute(0, 3, 4, 1, 2)
        
        return fused_feat


class RecurrentTemporalFusion(nn.Module):
    """Fuse temporal features using ConvLSTM.
    Maintain a hidden state that summarizes historical information.
    """
    def __init__(self, 
                 feature_dim: int = 256,
                 hidden_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # ConvLSTM for temporal fusion
        self.conv_lstm = ConvLSTM(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            kernel_size=3,
            num_layers=1
        )
        
        # 1x1 conv to match dimensions if needed
        self.out_conv = None
        if hidden_dim != feature_dim:
            self.out_conv = nn.Conv2d(hidden_dim, feature_dim, 1)
            
    def forward(self, features: torch.Tensor, hidden_state: Optional[tuple] = None
               ) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            features: Tensor[B, T, C, H, W]
            hidden_state: Optional tuple of (h, c) from previous step
        Returns:
            Tensor[B, T, C, H, W]: Temporally fused features
            tuple: Updated hidden state (h, c)
        """
        B, T, C, H, W = features.shape
        
        # Process sequence through ConvLSTM
        output, hidden = self.conv_lstm(features, hidden_state)
        
        # Match dimensions if needed
        if self.out_conv is not None:
            output = self.out_conv(output.view(-1, self.hidden_dim, H, W))
            output = output.view(B, T, -1, H, W)
            
        return output, hidden


class ConvLSTM(nn.Module):
    """Convolutional LSTM module."""
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, num_layers: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.padding = kernel_size // 2
        
        # Input gate
        self.conv_xi = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=self.padding)
        self.conv_hi = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=self.padding)
        
        # Forget gate
        self.conv_xf = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=self.padding)
        self.conv_hf = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=self.padding)
        
        # Cell gate
        self.conv_xc = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=self.padding)
        self.conv_hc = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=self.padding)
        
        # Output gate
        self.conv_xo = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=self.padding)
        self.conv_ho = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=self.padding)

    def forward(self, x: torch.Tensor, hidden: Optional[tuple] = None
               ) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            x: Input tensor [B, T, C, H, W]
            hidden: Optional tuple of (h, c) from previous step
        Returns:
            Tensor[B, T, C, H, W]: Output features
            tuple: Updated hidden state (h, c)
        """
        B, T, C, H, W = x.shape
        
        # Initialize hidden state if not provided
        if hidden is None:
            h = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
            c = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        else:
            h, c = hidden
            
        output = []
        
        # Process each time step
        for t in range(T):
            xt = x[:, t]
            
            # Input gate
            i = torch.sigmoid(self.conv_xi(xt) + self.conv_hi(h))
            
            # Forget gate
            f = torch.sigmoid(self.conv_xf(xt) + self.conv_hf(h))
            
            # Cell gate
            g = torch.tanh(self.conv_xc(xt) + self.conv_hc(h))
            
            # Output gate
            o = torch.sigmoid(self.conv_xo(xt) + self.conv_ho(h))
            
            # Update cell state
            c = f * c + i * g
            
            # Update hidden state
            h = o * torch.tanh(c)
            
            output.append(h)
            
        # Stack output sequence
        output = torch.stack(output, dim=1)
        
        return output, (h, c) 