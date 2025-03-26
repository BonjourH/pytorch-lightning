import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

from xinnovation.src.core.registry import MODEL
from xinnovation.src.components.lightning_module.transformer.transformer_layer import BaseTransformerLayer
from xinnovation.src.components.lightning_module.transformer.transformer_encoder import DetrTransformerEncoder
from xinnovation.src.components.lightning_module.transformer.transformer_decoder import DetrTransformerDecoder

@LIGHTNING_MODULE.register_module()
class DetrTransformer(nn.Module):
    """DETR Transformer.
    
    Args:
        encoder (dict): Encoder configuration.
        decoder (dict): Decoder configuration.
    """
    
    def __init__(
        self,
        encoder: Dict,
        decoder: Dict
    ):
        super().__init__()
        
        # Build encoder
        self.encoder = DetrTransformerEncoder(**encoder)
        
        # Build decoder
        self.decoder = DetrTransformerDecoder(**decoder)
        
    def forward(
        self,
        x: torch.Tensor,
        query_embed: torch.Tensor
    ) -> torch.Tensor:
        """Forward function.
        
        Args:
            x (torch.Tensor): Input features.
            query_embed (torch.Tensor): Query embeddings.
            
        Returns:
            torch.Tensor: Decoder outputs.
        """
        # Encoder forward
        memory = self.encoder(x)
        
        # Decoder forward
        hs = self.decoder(
            query_embed,
            memory,
            memory_pos=None,
            query_pos=None,
            memory_mask=None,
            query_mask=None,
            memory_key_padding_mask=None,
            query_key_padding_mask=None
        )
        
        return hs 