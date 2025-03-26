import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Any, List, Tuple
from abc import ABCMeta, abstractmethod
import torchviz
from torchinfo import summary
from pathlib import Path

__all__ = ["BaseDetector"]

class BaseDetector(nn.Module, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self):
        super(BaseDetector, self).__init__()
        
    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None
    
    @property
    def with_head(self):
        return hasattr(self, 'head') and self.head is not None
    
    @property
    def with_attention(self):
        return hasattr(self, 'attention') and self.attention is not None
    
    @property
    def with_norm_layer(self):
        return hasattr(self, 'norm_layer') and self.norm_layer is not None
    
    @property
    def with_pos_encoding(self):
        return hasattr(self, 'pos_encoding') and self.pos_encoding is not None
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
        
    def print_model_info(self) -> None:
        """打印模型结构信息"""
        print('Model Structure:')
        print('---------------')
        
        if hasattr(self, 'backbone'):
            print(f'Backbone: {self.backbone.__class__.__name__}')
            
        if self.with_neck:
            print(f'Neck: {self.neck.__class__.__name__}')
            
        if self.with_head:
            print(f'Head: {self.head.__class__.__name__}')
            
        if self.with_attention:
            print(f'Attention: {self.attention.__class__.__name__}')
            
        if self.with_norm_layer:
            print(f'Norm Layer: {self.norm_layer.__class__.__name__}')
            
        if self.with_pos_encoding:
            print(f'Position Encoding: {self.pos_encoding.__class__.__name__}')
            
        print('\nParameter Statistics:')
        print('--------------------')
        info = self.get_model_complexity_info()
        print(f'Total Parameters: {info["total_params"]:,}')
        print(f'Trainable Parameters: {info["trainable_params"]:,}')
        print(f'Model Size: {info["model_size"]:.2f} MB')
        print('\nDetailed Structure:')
        print('-----------------')
        print(info['summary_str'])
    
    def get_model_complexity_info(
        self,
        input_shape: Tuple[int, ...] = (3, 224, 224),
        batch_size: int = 1,
        device: str = 'cuda',
        **kwargs
    ) -> Dict[str, Any]:
        """计算模型的复杂度信息
        
        Args:
            input_shape (Tuple[int, ...]): 输入张量的形状 (C, H, W)
            batch_size (int): 批次大小
            device (str): 设备类型
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 包含以下信息：
                - total_params: 总参数量
                - trainable_params: 可训练参数量
                - total_flops: 总浮点运算量
                - model_size: 模型大小(MB)
                - summary_str: 模型结构摘要字符串
        """
        # 创建虚拟输入
        dummy_input = torch.randn(batch_size, *input_shape).to(device)
        self.to(device)
        
        # 使用torchinfo获取模型信息
        model_stats = summary(
            self,
            input_size=(batch_size, *input_shape),
            verbose=0,
            device=device,
            **kwargs
        )
        
        # 收集信息
        info = {
            'total_params': model_stats.total_params,
            'trainable_params': model_stats.trainable_params,
            'model_size': model_stats.total_params * 4 / (1024 * 1024),  # 假设每个参数4字节
            'summary_str': str(model_stats)
        }
        
        return info
    
    def visualize_model(
        self,
        input_shape: Tuple[int, ...] = (3, 224, 224),
        output_dir: Optional[str] = None,
        format: str = 'png',
        **kwargs
    ) -> None:
        """可视化模型结构
        
        Args:
            input_shape (Tuple[int, ...]): 输入张量的形状 (C, H, W)
            output_dir (str, optional): 输出目录
            format (str): 输出格式，支持'png'和'pdf'
            **kwargs: 其他参数传递给graphviz
        """
        # 创建虚拟输入
        dummy_input = torch.randn(1, *input_shape)
        
        # 生成计算图
        graph = torchviz.make_dot(
            self.forward(dummy_input),
            params=dict(self.named_parameters()),
            **kwargs
        )
        
        # 设置输出路径
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f'model_structure.{format}'
        else:
            output_path = f'model_structure.{format}'
            
        # 保存图像
        graph.render(str(output_path), format=format, cleanup=True)

