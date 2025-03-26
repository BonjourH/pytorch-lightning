import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

from xinnovation.src.core.registry import MODEL
from xinnovation.src.components.lightning_module.attention import MultiheadAttention, MultiScaleDeformableAttention

@LIGHTNING_MODULE.register_module()
class BaseTransformerLayer(nn.Module):
    """Base Transformer Layer.
    
    Args:
        attn_cfgs (dict): Attention configuration.
        feedforward_channels (int): Number of channels in feedforward network.
        ffn_dropout (float): Dropout rate in feedforward network.
        operation_order (tuple): Operation order in transformer layer.
    """
    
    def __init__(
        self,
        attn_cfgs: Dict,
        feedforward_channels: int,
        ffn_dropout: float = 0.1,
        operation_order: Tuple = ('self_attn', 'norm', 'ffn', 'norm')
    ):
        super().__init__()
        
        # Build attention module
        if attn_cfgs['type'] == 'MultiScaleDeformableAttention':
            self.attn = MultiScaleDeformableAttention(**attn_cfgs)
        else:
            self.attn = MultiheadAttention(**attn_cfgs)
            
        # Build feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(attn_cfgs['embed_dims'], feedforward_channels),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(feedforward_channels, attn_cfgs['embed_dims'])
        )
        
        # Build layer normalization
        self.norm1 = nn.LayerNorm(attn_cfgs['embed_dims'])
        self.norm2 = nn.LayerNorm(attn_cfgs['embed_dims'])
        
        # Operation order
        self.operation_order = operation_order
        
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
            torch.Tensor: Layer outputs.
        """
        # Forward through operations in specified order
        for op in self.operation_order:
            if op == 'self_attn':
                x = x + self.attn(
                    x,
                    x,
                    x,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    pos=pos
                )
            elif op == 'norm':
                x = self.norm1(x)
            elif op == 'ffn':
                x = x + self.ffn(x)
                x = self.norm2(x)
                
        return x

@LIGHTNING_MODULE.register_module()
class DetrTransformerDecoderLayer(nn.Module):
    """DETR Transformer Decoder Layer.
    
    Args:
        attn_cfgs (list): List of attention configurations.
        feedforward_channels (int): Number of channels in feedforward network.
        ffn_dropout (float): Dropout rate in feedforward network.
        operation_order (tuple): Operation order in transformer layer.
    """
    
    def __init__(
        self,
        attn_cfgs: List[Dict],
        feedforward_channels: int,
        ffn_dropout: float = 0.1,
        operation_order: Tuple = ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
    ):
        super().__init__()
        
        # Build self-attention module
        if attn_cfgs[0]['type'] == 'MultiScaleDeformableAttention':
            self.self_attn = MultiScaleDeformableAttention(**attn_cfgs[0])
        else:
            self.self_attn = MultiheadAttention(**attn_cfgs[0])
            
        # Build cross-attention module
        if attn_cfgs[1]['type'] == 'MultiScaleDeformableAttention':
            self.cross_attn = MultiScaleDeformableAttention(**attn_cfgs[1])
        else:
            self.cross_attn = MultiheadAttention(**attn_cfgs[1])
            
        # Build feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(attn_cfgs[0]['embed_dims'], feedforward_channels),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(feedforward_channels, attn_cfgs[0]['embed_dims'])
        )
        
        # Build layer normalization
        self.norm1 = nn.LayerNorm(attn_cfgs[0]['embed_dims'])
        self.norm2 = nn.LayerNorm(attn_cfgs[0]['embed_dims'])
        self.norm3 = nn.LayerNorm(attn_cfgs[0]['embed_dims'])
        
        # Operation order
        self.operation_order = operation_order
        
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function.
        
        Args:
            tgt (torch.Tensor): Target sequence.
            memory (torch.Tensor): Memory sequence.
            tgt_mask (torch.Tensor, optional): Target attention mask.
            memory_mask (torch.Tensor, optional): Memory attention mask.
            tgt_key_padding_mask (torch.Tensor, optional): Target key padding mask.
            memory_key_padding_mask (torch.Tensor, optional): Memory key padding mask.
            pos (torch.Tensor, optional): Positional encoding for memory.
            query_pos (torch.Tensor, optional): Positional encoding for queries.
            
        Returns:
            torch.Tensor: Layer outputs.
        """
        # Forward through operations in specified order
        for op in self.operation_order:
            if op == 'self_attn':
                tgt = tgt + self.self_attn(
                    tgt + query_pos,
                    tgt + query_pos,
                    tgt,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )
                tgt = self.norm1(tgt)
            elif op == 'cross_attn':
                tgt = tgt + self.cross_attn(
                    tgt + query_pos,
                    memory + pos,
                    memory,
                    attn_mask=memory_mask,
                    key_padding_mask=memory_key_padding_mask
                )
                tgt = self.norm2(tgt)
            elif op == 'ffn':
                tgt = tgt + self.ffn(tgt)
                tgt = self.norm3(tgt)
                
        return tgt 