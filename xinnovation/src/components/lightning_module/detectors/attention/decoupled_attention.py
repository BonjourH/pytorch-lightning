import torch
import torch.nn as nn
from xinnovation.src.core.registry import ATTENTION, NORM_LAYERS
from typing import Optional, Dict
from xinnovation.src.core import build_from_cfg

__all__ = ["DecoupledMultiHeadAttention"]


@ATTENTION.register_module()
class DecoupledMultiHeadAttention(nn.Module):
    """
    Decoupled multi-head attention.
    """
    def __init__(self, query_dim: int = 256, num_heads: int = 8, dropout: float = 0.1, post_norm: Dict = None):
        super().__init__()
        self.value_linear = nn.Linear(query_dim, query_dim * 2, bias=False)
        self.query_linear = nn.Linear(query_dim * 2, query_dim, bias=False)
        self.attn = nn.MultiheadAttention(query_dim * 2, num_heads, batch_first=True, dropout=dropout)
        self.post_norm = build_from_cfg(post_norm, NORM_LAYERS)
        
        
    def init_weights(self):
        # 使用Xavier/Glorot初始化线性层
        nn.init.xavier_normal_(self.value_linear.weight.data)
        nn.init.xavier_normal_(self.query_linear.weight.data)
        
        # 注意力机制的投影权重初始化
        nn.init.xavier_normal_(self.attn.in_proj_weight.data)
        nn.init.xavier_normal_(self.attn.out_proj.weight.data)
        
        # 如果有偏置项，对其进行初始化（假设注意力机制有偏置）
        if hasattr(self.attn, 'in_proj_bias') and self.attn.in_proj_bias is not None:
            nn.init.zeros_(self.attn.in_proj_bias)
        if hasattr(self.attn, 'out_proj') and hasattr(self.attn.out_proj, 'bias') and self.attn.out_proj.bias is not None:
            nn.init.zeros_(self.attn.out_proj.bias)
        
        # 对归一化层使用更适合的初始化方法
        if self.post_norm is not None:
            self.post_norm.init_weights()

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
        if self.post_norm is not None:
            tgt = self.post_norm(tgt)
        return tgt

