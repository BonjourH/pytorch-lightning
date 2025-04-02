"""智能构建工具，用于从配置构建各种组件"""

from typing import Dict, Any, Optional, Union, TypeVar, Type
from .registry import Registry, LIGHTNING_MODULE, DATA, TRAINER, OPTIMIZERS, SCHEDULERS

from omegaconf import DictConfig
from .config import Config
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer


def build_from_cfg(cfg: Dict[str, Any], registry: Registry) -> Any:
    """从配置构建组件
    
    Args:
        cfg (Dict[str, Any]): 组件配置
        registry (Registry): 组件注册表
    """
    if cfg is None:
        return None
    return registry.build(cfg)

def build_model(cfg: Dict[str, Any], **kwargs) -> Any:
    """构建模型组件
    
    Args:
        cfg (Dict[str, Any]): 模型配置
        **kwargs: 额外参数
        
    Returns:
        Any: 构建的模型组件
    """
    return LIGHTNING_MODULE.build(cfg, **kwargs)

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

def build_optimizer(cfg: Dict[str, Any], params: Optional[torch.nn.Parameter] = None, **kwargs) -> Optimizer:
    """根据配置构建优化器
    
    Args:
        cfg (Dict[str, Any]): 优化器配置字典
        params (Optional[torch.nn.Parameter]): 需要优化的参数
        **kwargs: 传递给优化器的额外参数
        
    Returns:
        Optimizer: 构建的优化器实例
        
    Examples:
        >>> # 使用配置字典构建
        >>> optimizer_cfg = dict(
        ...     type='SGDOptimizer',
        ...     lr=0.01,
        ...     momentum=0.9,
        ...     weight_decay=0.0001
        ... )
        >>> optimizer = build_optimizer(optimizer_cfg, params=model.parameters())
    """
    # 检查是否是直接传入的优化器实例（通过_convert_to_config_dict转换后的）
    if "_instance" in cfg:
        return cfg["_instance"]
        
    cfg = cfg.copy()
    optimizer_type = cfg.pop('type')
    
    # 如果params未提供，尝试从kwargs中获取
    if params is None and 'params' in kwargs:
        params = kwargs.pop('params')
        
    if params is None:
        raise ValueError('params must be specified')
        
    # 构建优化器
    optimizer_wrapper = OPTIMIZERS.build(
        dict(type=optimizer_type, **cfg),
        params=params,
        **kwargs
    )
    
    # 返回优化器包装类中的optimizer属性
    return optimizer_wrapper.optimizer


def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer, **kwargs) -> _LRScheduler:
    """根据配置构建学习率调度器
    
    Args:
        cfg (Dict[str, Any]): 调度器配置字典
        optimizer (Optimizer): 优化器实例
        **kwargs: 传递给调度器的额外参数
        
    Returns:
        _LRScheduler: 构建的学习率调度器实例
        
    Examples:
        >>> # 使用配置字典构建
        >>> scheduler_cfg = dict(
        ...     type='StepLRScheduler',
        ...     step_size=10,
        ...     gamma=0.1
        ... )
        >>> scheduler = build_scheduler(scheduler_cfg, optimizer=optimizer)
    """
    # 检查是否是直接传入的调度器实例（通过_convert_to_config_dict转换后的）
    if "_instance" in cfg:
        return cfg["_instance"]
        
    cfg = cfg.copy()
    scheduler_type = cfg.pop('type')
    
    # 构建调度器
    scheduler_wrapper = SCHEDULERS.build(
        dict(type=scheduler_type, **cfg),
        optimizer=optimizer,
        **kwargs
    )
    
    # 返回调度器包装类中的scheduler属性
    return scheduler_wrapper.scheduler
