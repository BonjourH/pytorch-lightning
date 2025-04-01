from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import parallel_apply


def grid_sample_fpn_features(fpn_features: List[torch.Tensor], 
                             pixels: torch.Tensor,
                             mode: str = 'bilinear',
                             padding_mode: str = 'zeros',
                             align_corners: bool = True) -> List[torch.Tensor]:
    """
    从多层FPN特征中采样像素对应的特征
    
    Args:
        fpn_features: 不同层级的特征图列表，每个元素形状为 [B, C, H_i, W_i]
        pixels: 归一化的像素坐标，范围在[-1, 1]之间，形状为 [B, N, 2]
        
    Returns:
        feature_list: List[torch.Tensor], 每个元素形状为 [B, C, H_out, W_out]
    """
    if pixels.ndim == 3:
        pixels = pixels.unsqueeze(2)
    
    # 准备存储所有采样结果的张量
    sampled_features = []
    
    # 对每个FPN层级进行采样
    for features in fpn_features:
        
        # 使用grid_sample采样特征
        # features: [B, C, H, W], grid: [B, H_out, W_out, 2] -> sampled: [B, C, H_out, W_out]
        sampled = F.grid_sample(
            features, 
            pixels, 
            mode=mode, 
            padding_mode=padding_mode, 
            align_corners=align_corners
        )
        
        sampled_features.append(sampled)
        
    return torch.stack(sampled_features, dim=1)


def grid_sample_fpn_features_parallel_apply(fpn_features: List[torch.Tensor], 
                                            pixels: torch.Tensor,
                                            mode: str = 'bilinear',
                                            padding_mode: str = 'zeros',
                                            align_corners: bool = True,
                                            dim: int = -1) -> torch.Tensor:
    """
    使用parallel_apply从多层FPN特征中并行采样
    """
    if pixels.ndim == 3:
        pixels = pixels.unsqueeze(2)
    
    # 定义对单层特征的采样函数
    def sample_fn(feature):
        return F.grid_sample(
            feature, 
            pixels, 
            mode=mode, 
            padding_mode=padding_mode, 
            align_corners=align_corners
        )
    
    # 创建采样模块列表
    modules = [lambda f=feature: sample_fn(f) for feature in fpn_features]
    
    # 并行应用函数到每个特征图 - 使用空列表作为输入，因为模块内部已包含要处理的数据
    results = parallel_apply(modules, inputs=[[] for _ in range(len(modules))])
    
    return torch.stack(results, dim=dim)

# TODO: define a cuda kernel for grid_sample from different scales fpn_features


if __name__ == "__main__":
    fpn_features = [torch.randn(1, 256, 16, 16), torch.randn(1, 256, 32, 32)]
    pixels = torch.randn(1, 10, 2)
    sampled_features = grid_sample_fpn_features(fpn_features, pixels)
    print(sampled_features.shape)
    
    fpn_features = [torch.randn(2, 256, 16, 16), torch.randn(2, 256, 32, 32), torch.randn(2, 256, 64, 64)]
    pixels = torch.randn(2, 10, 2)
    
    sampled_features_parallel_apply = grid_sample_fpn_features_parallel_apply(fpn_features, pixels)
    print(sampled_features_parallel_apply.shape)
    
    
    fpn_features = [torch.randn(2, 256, 16, 16), torch.randn(2, 256, 32, 32), torch.randn(2, 256, 64, 64)]
    pixels = torch.randn(2, 10, 10, 2)
    
    sampled_features_parallel_apply = grid_sample_fpn_features_parallel_apply(fpn_features, pixels)
    print(sampled_features_parallel_apply.shape)