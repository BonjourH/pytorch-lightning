import torch
import torch.nn as nn
from xinnovation.src.core.registry import ATTENTION
from typing import Optional

__all__ = ["DecoupledMultiHeadAttention"]


@ATTENTION.register_module()
class DecoupledMultiHeadAttention(nn.Module):
    """
    Decoupled multi-head attention.
    """
    def __init__(self, query_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.value_linear = nn.Linear(query_dim, query_dim * 2, bias=False)
        self.query_linear = nn.Linear(query_dim * 2, query_dim, bias=False)
        self.attn = nn.MultiheadAttention(query_dim, num_heads, batch_first=True, dropout=dropout)

    def forward(self, tgt: torch.Tensor, 
                pos_tgt: torch.Tensor, 
                memory: Optional[torch.Tensor]=None, 
                pos_memory: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Args:
            tgt: Tensor[B, N, query_dim], current query
            pos_tgt: Tensor[B, N, query_dim], current query position encoding
            memory: Tensor[B, M, query_dim], tracked query
            pos_memory: Tensor[B, M, query_dim], tracked query position encoding

        Returns:
            Tensor[B, N, query_dim]
        """
        memory = memory if memory is not None else tgt
        pos_memory = pos_memory if pos_memory is not None else pos_tgt

        v = self.value_linear(memory) # [B, M, 2 * query_dim]

        tgt = torch.cat([tgt, pos_tgt], dim=2) # [B, N, 2 * query_dim]
        memory = torch.cat([memory, pos_memory], dim=2) # [B, M, 2 * query_dim]

        tgt = self.attn(tgt, memory, v)[0] # [B, N, 2 * query_dim]
        tgt = self.query_linear(tgt) # [B, N, query_dim]
        return tgt

