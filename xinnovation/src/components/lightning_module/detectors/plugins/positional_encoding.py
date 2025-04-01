import torch
import torch.nn as nn
from typing import Dict, Optional

from xinnovation.src.core.registry import POS_ENCODINGS

@POS_ENCODINGS.register_module()
class SinePositionalEncoding(nn.Module):
    """Sine Positional Encoding.
    
    Args:
        num_feats (int): Number of features.
        temperature (float): Temperature for scaling.
        normalize (bool): Whether to normalize the encoding.
        scale (float): Scale factor for the encoding.
    """
    
    def __init__(
        self,
        num_feats: int,
        temperature: float = 10000,
        normalize: bool = False,
        scale: float = 2 * 3.14159265359
    ):
        super().__init__()
        
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Positional encoding.
        """
        batch_size, _, height, width = x.size()
        
        # Generate position indices
        y_embed = torch.arange(height, dtype=torch.float32, device=x.device)
        x_embed = torch.arange(width, dtype=torch.float32, device=x.device)
        
        # Normalize if required
        if self.normalize:
            y_embed = y_embed / (height - 1) * self.scale
            x_embed = x_embed / (width - 1) * self.scale
        else:
            y_embed = y_embed / height * self.scale
            x_embed = x_embed / width * self.scale
            
        # Generate position embeddings
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        
        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=1).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=1).flatten(1)
        
        # Combine position embeddings
        pos = torch.cat((pos_y, pos_x), dim=1)
        pos = pos.unsqueeze(0).expand(batch_size, -1, -1)
        
        return pos 