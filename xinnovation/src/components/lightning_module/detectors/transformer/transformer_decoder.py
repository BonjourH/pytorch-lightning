import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

from xinnovation.src.core.registry import MODEL
from xinnovation.src.components.lightning_module.transformer.transformer_layer import DetrTransformerDecoderLayer

@LIGHTNING_MODULE.register_module()
class DetrTransformerDecoder(nn.Module):
    """DETR Transformer Decoder.
    
    Args:
        num_layers (int): Number of decoder layers.
        transformerlayers (dict): Transformer layer configuration.
        return_intermediate (bool): Whether to return intermediate outputs.
    """
    
    def __init__(
        self,
        num_layers: int,
        transformerlayers: Dict,
        return_intermediate: bool = True
    ):
        super().__init__()
        
        # Build decoder layers
        self.layers = nn.ModuleList([
            DetrTransformerDecoderLayer(**transformerlayers)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(transformerlayers['attn_cfgs'][0]['embed_dims'])
        
        # Whether to return intermediate outputs
        self.return_intermediate = return_intermediate
        
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
            torch.Tensor: Decoder outputs.
        """
        output = tgt
        intermediate = []
        
        # Forward through decoder layers
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos
            )
            
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                
        # Layer normalization
        output = self.norm(output)
        
        if self.return_intermediate:
            intermediate.append(output)
            return torch.stack(intermediate)
            
        return output 