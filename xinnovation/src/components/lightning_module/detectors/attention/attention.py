import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from xinnovation.src.core.registry import ATTENTION

@ATTENTION.register_module()
class MultiheadAttention(nn.Module):
    """Multi-head Attention.
    
    Args:
        embed_dims (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
    """
    
    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Check if embed_dims is divisible by num_heads
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})")
            
        self.head_dims = embed_dims // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function.
        
        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            attn_mask (torch.Tensor, optional): Attention mask.
            key_padding_mask (torch.Tensor, optional): Key padding mask.
            pos (torch.Tensor, optional): Positional encoding.
            
        Returns:
            torch.Tensor: Attention output.
        """
        batch_size = query.size(0)
        
        # Linear projections
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dims).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dims).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dims).transpose(1, 2)
        
        # Add positional encoding if provided
        if pos is not None:
            k = k + pos.view(batch_size, -1, self.num_heads, self.head_dims).transpose(1, 2)
            
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dims ** 0.5)
        
        # Apply attention mask
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))
            
        # Apply key padding mask
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
            
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dims)
        out = self.out_proj(out)
        
        return out

@ATTENTION.register_module()
class MultiScaleDeformableAttention(nn.Module):
    """Multi-scale Deformable Attention.
    
    Args:
        embed_dims (int): Embedding dimension.
        num_levels (int): Number of feature levels.
        num_heads (int): Number of attention heads.
        num_points (int): Number of sampling points.
        dropout (float): Dropout rate.
    """
    
    def __init__(
        self,
        embed_dims: int,
        num_levels: int = 4,
        num_heads: int = 8,
        num_points: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.dropout = dropout
        
        # Check if embed_dims is divisible by num_heads
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})")
            
        self.head_dims = embed_dims // num_heads
        
        # Linear projections
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function.
        
        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            attn_mask (torch.Tensor, optional): Attention mask.
            key_padding_mask (torch.Tensor, optional): Key padding mask.
            pos (torch.Tensor, optional): Positional encoding.
            
        Returns:
            torch.Tensor: Attention output.
        """
        batch_size = query.size(0)
        
        # Project value
        value = self.value_proj(value)
        value = value.view(batch_size, -1, self.num_heads, self.head_dims)
        
        # Generate sampling offsets
        sampling_offsets = self.sampling_offsets(query).view(
            batch_size, -1, self.num_heads, self.num_levels, self.num_points, 2
        )
        
        # Generate attention weights
        attention_weights = self.attention_weights(query).view(
            batch_size, -1, self.num_heads, self.num_levels, self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Add positional encoding if provided
        if pos is not None:
            sampling_offsets = sampling_offsets + pos.view(
                batch_size, -1, self.num_heads, self.num_levels, self.num_points, 2
            )
            
        # Apply key padding mask
        if key_padding_mask is not None:
            value = value.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                0
            )
            
        # Sample values
        output = torch.zeros(
            batch_size,
            query.size(1),
            self.embed_dims,
            device=query.device,
            dtype=query.dtype
        )
        
        for level in range(self.num_levels):
            level_value = value[:, level]
            level_offset = sampling_offsets[:, :, :, level]
            level_weight = attention_weights[:, :, :, level]
            
            # Apply sampling
            sampled_value = F.grid_sample(
                level_value.permute(0, 2, 3, 1),
                level_offset,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )
            
            # Weighted sum
            output = output + torch.matmul(
                level_weight.unsqueeze(-1),
                sampled_value
            ).squeeze(-1)
            
        # Project output
        output = self.output_proj(output)
        
        return output 