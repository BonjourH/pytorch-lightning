import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

from xinnovation.src.core.registry import MODEL
from xinnovation.src.components.lightning_module.transformer.transformer_layer import BaseTransformerLayer

@LIGHTNING_MODULE.register_module()
class DetrTransformerEncoder(nn.Module):
    """DETR Transformer Encoder.
    
    Args:
        num_layers (int): Number of encoder layers.
        transformerlayers (dict): Transformer layer configuration.
    """
    
    def __init__(
        self,
        num_layers: int,
        transformerlayers: Dict
    ):
        super().__init__()
        
        # Build encoder layers
        self.layers = nn.ModuleList([
            BaseTransformerLayer(**transformerlayers)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(transformerlayers['attn_cfgs']['embed_dims'])
        
    def forward(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function.
        
        Args:
            x (torch.Tensor): Input features.
            pos (torch.Tensor, optional): Positional encoding.
            attn_mask (torch.Tensor, optional): Attention mask.
            key_padding_mask (torch.Tensor, optional): Key padding mask.
            
        Returns:
            torch.Tensor: Encoder outputs.
        """
        # Forward through encoder layers
        for layer in self.layers:
            x = layer(
                x,
                pos=pos,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask
            )
            
        # Layer normalization
        x = self.norm(x)
        
        return x 