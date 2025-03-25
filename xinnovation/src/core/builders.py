"""智能构建工具，用于从配置构建各种组件"""

from typing import Dict, Any, Optional
from .registry import MODEL, DATA, TRAINER

def build_model(cfg: Dict[str, Any], **kwargs) -> Any:
    """构建模型组件
    
    Args:
        cfg (Dict[str, Any]): 模型配置
        **kwargs: 额外参数
        
    Returns:
        Any: 构建的模型组件
    """
    return MODEL.build(cfg, **kwargs)

def build_dataset(cfg: Dict[str, Any], **kwargs) -> Any:
    """构建数据集组件
    
    Args:
        cfg (Dict[str, Any]): 数据集配置
        **kwargs: 额外参数
        
    Returns:
        Any: 构建的数据集组件
    """
    return DATA.build(cfg, **kwargs)

def build_trainer(cfg: Dict[str, Any], **kwargs) -> Any:
    """构建训练器组件
    
    Args:
        cfg (Dict[str, Any]): 训练器配置
        **kwargs: 额外参数
        
    Returns:
        Any: 构建的训练器组件
    """
    return TRAINER.build(cfg, **kwargs) 