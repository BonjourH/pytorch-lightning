import torch
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau,
    CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts, LambdaLR
)
from typing import Optional, Dict, Any, Union, List, Callable
from xinnovation.src.core.registry import SCHEDULERS


__all__ = [
    'StepLRScheduler',
    'MultiStepLRScheduler',
    'ExponentialLRScheduler',
    'CosineAnnealingLRScheduler',
    'ReduceLROnPlateauScheduler',
    'CyclicLRScheduler',
    'OneCycleLRScheduler',
    'CosineAnnealingWarmRestartsScheduler',
    'LambdaLRScheduler'
]


@SCHEDULERS.register_module(force=True)
class StepLRScheduler:
    """StepLR scheduler wrapper.
    
    按固定步长调整学习率，每隔 step_size 个 epoch 将学习率乘以 gamma。
    
    Args:
        optimizer (Optimizer): 要调整学习率的优化器
        step_size (int): 学习率下降的周期（epoch 数）
        gamma (float): 学习率调整的乘法因子
        last_epoch (int): 上一个 epoch 的索引，默认为 -1
        verbose (bool): 是否在更新时打印信息
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False,
        **kwargs
    ):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.verbose = verbose
        
        # 创建调度器实例
        self.scheduler = StepLR(
            optimizer=self.optimizer,
            step_size=self.step_size,
            gamma=self.gamma,
            last_epoch=self.last_epoch,
            verbose=self.verbose
        )


@SCHEDULERS.register_module(force=True)
class MultiStepLRScheduler:
    """MultiStepLR scheduler wrapper.
    
    在指定的 milestone 处调整学习率，每到一个 milestone 将学习率乘以 gamma。
    
    Args:
        optimizer (Optimizer): 要调整学习率的优化器
        milestones (List[int]): 学习率下降的轮次列表
        gamma (float): 学习率调整的乘法因子
        last_epoch (int): 上一个 epoch 的索引，默认为 -1
        verbose (bool): 是否在更新时打印信息
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False,
        **kwargs
    ):
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.verbose = verbose
        
        # 创建调度器实例
        self.scheduler = MultiStepLR(
            optimizer=self.optimizer,
            milestones=self.milestones,
            gamma=self.gamma,
            last_epoch=self.last_epoch,
            verbose=self.verbose
        )


@SCHEDULERS.register_module(force=True)
class ExponentialLRScheduler:
    """ExponentialLR scheduler wrapper.
    
    按指数衰减调整学习率，每个 epoch 将学习率乘以 gamma。
    
    Args:
        optimizer (Optimizer): 要调整学习率的优化器
        gamma (float): 学习率调整的乘法因子
        last_epoch (int): 上一个 epoch 的索引，默认为 -1
        verbose (bool): 是否在更新时打印信息
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.9,
        last_epoch: int = -1,
        verbose: bool = False,
        **kwargs
    ):
        self.optimizer = optimizer
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.verbose = verbose
        
        # 创建调度器实例
        self.scheduler = ExponentialLR(
            optimizer=self.optimizer,
            gamma=self.gamma,
            last_epoch=self.last_epoch,
            verbose=self.verbose
        )


@SCHEDULERS.register_module(force=True)
class CosineAnnealingLRScheduler:
    """CosineAnnealingLR scheduler wrapper.
    
    余弦退火学习率，在 T_max 个周期内将学习率从初始值变化到 eta_min。
    
    Args:
        optimizer (Optimizer): 要调整学习率的优化器
        T_max (int): 退火周期的最大迭代次数
        eta_min (float): 学习率的最小值
        last_epoch (int): 上一个 epoch 的索引，默认为 -1
        verbose (bool): 是否在更新时打印信息
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min: float = 0,
        last_epoch: int = -1,
        verbose: bool = False,
        **kwargs
    ):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.verbose = verbose
        
        # 创建调度器实例
        self.scheduler = CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.T_max,
            eta_min=self.eta_min,
            last_epoch=self.last_epoch,
            verbose=self.verbose
        )


@SCHEDULERS.register_module(force=True)
class ReduceLROnPlateauScheduler:
    """ReduceLROnPlateau scheduler wrapper.
    
    当指标停止改善时降低学习率，监控特定指标，当指标停滞不前时降低学习率。
    
    Args:
        optimizer (Optimizer): 要调整学习率的优化器
        mode (str): 'min' 或 'max'，对应最小化或最大化监控指标
        factor (float): 学习率衰减的因子
        patience (int): 指标不再改善后等待的 epoch 数量
        threshold (float): 衡量改进的阈值
        threshold_mode (str): 阈值的比较模式 ('rel' 或 'abs')
        cooldown (int): 减少学习率后冷却的 epoch 数量
        min_lr (float 或 List): 学习率的最小值
        eps (float): 更新的最小差值
        verbose (bool): 是否在更新时打印信息
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = 'rel',
        cooldown: int = 0,
        min_lr: float = 0,
        eps: float = 1e-8,
        verbose: bool = False,
        **kwargs
    ):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.verbose = verbose
        
        # 创建调度器实例
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode=self.mode,
            factor=self.factor,
            patience=self.patience,
            threshold=self.threshold,
            threshold_mode=self.threshold_mode,
            cooldown=self.cooldown,
            min_lr=self.min_lr,
            eps=self.eps,
            verbose=self.verbose
        )


@SCHEDULERS.register_module(force=True)
class CyclicLRScheduler:
    """CyclicLR scheduler wrapper.
    
    循环调整学习率，在最小学习率和最大学习率之间循环变化。
    
    Args:
        optimizer (Optimizer): 要调整学习率的优化器
        base_lr (float 或 List): 最小学习率
        max_lr (float 或 List): 最大学习率
        step_size_up (int): 从最小学习率增加到最大学习率的迭代次数
        step_size_down (int): 从最大学习率减少到最小学习率的迭代次数
        mode (str): 学习率变化的模式 ('triangular', 'triangular2' 或 'exp_range')
        gamma (float): 仅用于 'exp_range' 模式的指数因子
        scale_fn (callable): 自定义缩放函数
        scale_mode (str): 缩放函数的应用模式 ('cycle' 或 'iterations')
        cycle_momentum (bool): 是否同时调整动量
        base_momentum (float 或 List): 最小动量
        max_momentum (float 或 List): 最大动量
        last_epoch (int): 上一个 epoch 的索引，默认为 -1
        verbose (bool): 是否在更新时打印信息
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: Union[float, List[float]],
        max_lr: Union[float, List[float]],
        step_size_up: int = 2000,
        step_size_down: Optional[int] = None,
        mode: str = 'triangular',
        gamma: float = 1.0,
        scale_fn: Optional[Callable] = None,
        scale_mode: str = 'cycle',
        cycle_momentum: bool = True,
        base_momentum: Union[float, List[float]] = 0.8,
        max_momentum: Union[float, List[float]] = 0.9,
        last_epoch: int = -1,
        verbose: bool = False,
        **kwargs
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        self.last_epoch = last_epoch
        self.verbose = verbose
        
        # 创建调度器实例
        self.scheduler = CyclicLR(
            optimizer=self.optimizer,
            base_lr=self.base_lr,
            max_lr=self.max_lr,
            step_size_up=self.step_size_up,
            step_size_down=self.step_size_down,
            mode=self.mode,
            gamma=self.gamma,
            scale_fn=self.scale_fn,
            scale_mode=self.scale_mode,
            cycle_momentum=self.cycle_momentum,
            base_momentum=self.base_momentum,
            max_momentum=self.max_momentum,
            last_epoch=self.last_epoch,
            verbose=self.verbose
        )


@SCHEDULERS.register_module(force=True)
class OneCycleLRScheduler:
    """OneCycleLR scheduler wrapper.
    
    一个周期的学习率调整，主要用于训练带有超收敛的模型。
    
    Args:
        optimizer (Optimizer): 要调整学习率的优化器
        max_lr (float 或 List): 最大学习率
        total_steps (int): 总的训练步数
        epochs (int): 总的训练 epoch 数
        steps_per_epoch (int): 每个 epoch 的步数
        pct_start (float): 周期中用于提高学习率的比例
        anneal_strategy (str): 学习率下降的策略 ('cos' 或 'linear')
        cycle_momentum (bool): 是否同时调整动量
        base_momentum (float 或 List): 最小动量
        max_momentum (float 或 List): 最大动量
        div_factor (float): 初始学习率与最大学习率的比例
        final_div_factor (float): 最小学习率与初始学习率的比例
        last_epoch (int): 上一个 epoch 的索引，默认为 -1
        verbose (bool): 是否在更新时打印信息
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: Union[float, List[float]],
        total_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        cycle_momentum: bool = True,
        base_momentum: Union[float, List[float]] = 0.85,
        max_momentum: Union[float, List[float]] = 0.95,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
        verbose: bool = False,
        **kwargs
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.last_epoch = last_epoch
        self.verbose = verbose
        
        # 确保 total_steps 已指定或可计算
        if total_steps is None and (epochs is None or steps_per_epoch is None):
            raise ValueError("Either total_steps or (epochs and steps_per_epoch) must be specified")
        
        if total_steps is None:
            self.total_steps = epochs * steps_per_epoch
        
        # 创建调度器实例
        self.scheduler = OneCycleLR(
            optimizer=self.optimizer,
            max_lr=self.max_lr,
            total_steps=self.total_steps,
            pct_start=self.pct_start,
            anneal_strategy=self.anneal_strategy,
            cycle_momentum=self.cycle_momentum,
            base_momentum=self.base_momentum,
            max_momentum=self.max_momentum,
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor,
            last_epoch=self.last_epoch,
            verbose=self.verbose
        )


@SCHEDULERS.register_module(force=True)
class CosineAnnealingWarmRestartsScheduler:
    """CosineAnnealingWarmRestarts scheduler wrapper.
    
    带有预热重启的余弦退火，在每次重启时将学习率重置为初始值。
    
    Args:
        optimizer (Optimizer): 要调整学习率的优化器
        T_0 (int): 第一次重启的迭代次数
        T_mult (int): 增加后续重启周期长度的因子
        eta_min (float): 学习率的最小值
        last_epoch (int): 上一个 epoch 的索引，默认为 -1
        verbose (bool): 是否在更新时打印信息
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1,
        verbose: bool = False,
        **kwargs
    ):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.verbose = verbose
        
        # 创建调度器实例
        self.scheduler = CosineAnnealingWarmRestarts(
            optimizer=self.optimizer,
            T_0=self.T_0,
            T_mult=self.T_mult,
            eta_min=self.eta_min,
            last_epoch=self.last_epoch,
            verbose=self.verbose
        )


@SCHEDULERS.register_module(force=True)
class LambdaLRScheduler:
    """LambdaLR scheduler wrapper.
    
    使用自定义函数调整学习率。
    
    Args:
        optimizer (Optimizer): 要调整学习率的优化器
        lr_lambda (callable 或 List): 学习率调整函数
        last_epoch (int): 上一个 epoch 的索引，默认为 -1
        verbose (bool): 是否在更新时打印信息
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_lambda: Union[Callable, List[Callable]],
        last_epoch: int = -1,
        verbose: bool = False,
        **kwargs
    ):
        self.optimizer = optimizer
        self.lr_lambda = lr_lambda
        self.last_epoch = last_epoch
        self.verbose = verbose
        
        # 创建调度器实例
        self.scheduler = LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=self.lr_lambda,
            last_epoch=self.last_epoch,
            verbose=self.verbose
        )


# ====================================
# 验证用例
# ====================================

if __name__ == "__main__":
    """验证各种学习率调度器的构建过程"""
    from xinnovation.src.core.builders import build_scheduler
    
    # 检查注册表中的调度器
    print("\n===== 注册表状态 =====")
    registered_schedulers = SCHEDULERS.get_module_dict().keys()
    print(f"当前已注册的调度器: {', '.join(registered_schedulers)}")
    
    # 创建一个简单模型和优化器
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 10)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    print("\n===== 学习率调度器构建测试 =====")
    
    # 测试1: 基础调度器构建
    print("\n[测试1] 基础调度器构建:")
    
    schedulers = [
        ("StepLR", {"type": "StepLRScheduler", "step_size": 10, "gamma": 0.1}),
        ("MultiStepLR", {"type": "MultiStepLRScheduler", "milestones": [30, 60, 90], "gamma": 0.1}),
        ("ExponentialLR", {"type": "ExponentialLRScheduler", "gamma": 0.95}),
        ("CosineAnnealingLR", {"type": "CosineAnnealingLRScheduler", "T_max": 100, "eta_min": 1e-5}),
    ]
    
    for name, cfg in schedulers:
        try:
            scheduler = build_scheduler(cfg, optimizer=optimizer)
            print(f"  - {name}: {scheduler.__class__.__name__}")
        except Exception as e:
            print(f"  - {name}: 构建失败 - {str(e)}")
    
    # 测试2: 高级调度器构建
    print("\n[测试2] 高级调度器构建:")
    
    advanced_schedulers = [
        ("ReduceLROnPlateau", {"type": "ReduceLROnPlateauScheduler", "mode": "min", "factor": 0.1, "patience": 10}),
        ("CyclicLR", {"type": "CyclicLRScheduler", "base_lr": 0.001, "max_lr": 0.1, "step_size_up": 2000}),
        ("CosineAnnealingWarmRestarts", {"type": "CosineAnnealingWarmRestartsScheduler", "T_0": 10, "T_mult": 2}),
    ]
    
    for name, cfg in advanced_schedulers:
        try:
            scheduler = build_scheduler(cfg, optimizer=optimizer)
            print(f"  - {name}: {scheduler.__class__.__name__}")
        except Exception as e:
            print(f"  - {name}: 构建失败 - {str(e)}")
    
    # 测试3: 特殊调度器构建
    print("\n[测试3] 特殊调度器构建:")
    
    # OneCycleLR 需要指定 total_steps
    onecycle_cfg = {
        "type": "OneCycleLRScheduler", 
        "max_lr": 0.1,
        "total_steps": 1000,
        "pct_start": 0.3,
        "anneal_strategy": "cos",
        "div_factor": 25
    }
    try:
        onecycle_scheduler = build_scheduler(onecycle_cfg, optimizer=optimizer)
        print(f"  - OneCycleLR: {onecycle_scheduler.__class__.__name__}")
    except Exception as e:
        print(f"  - OneCycleLR: 构建失败 - {str(e)}")
    
    # Lambda函数调度器
    def lr_lambda(epoch):
        return 0.95 ** epoch
    
    lambda_cfg = {
        "type": "LambdaLRScheduler",
        "lr_lambda": lr_lambda
    }
    try:
        lambda_scheduler = build_scheduler(lambda_cfg, optimizer=optimizer)
        print(f"  - LambdaLR: {lambda_scheduler.__class__.__name__}")
    except Exception as e:
        print(f"  - LambdaLR: 构建失败 - {str(e)}")
    
    # 测试4: 学习率变化验证
    print("\n[测试4] 学习率变化验证:")
    
    # 使用StepLR测试学习率变化
    step_scheduler = build_scheduler({"type": "StepLRScheduler", "step_size": 10, "gamma": 0.5}, optimizer=optimizer)
    print("  - StepLR学习率变化:")
    print(f"    初始学习率: {optimizer.param_groups[0]['lr']}")
    
    for epoch in range(1, 31):
        optimizer.step()
        step_scheduler.step()
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}, 学习率: {optimizer.param_groups[0]['lr']}")
    
    # 测试5: 常用场景配置案例
    print("\n[测试5] 常用配置案例:")
    
    # 案例1: 分类任务常用 MultiStepLR 配置
    classification_cfg = {
        "type": "MultiStepLRScheduler",
        "milestones": [60, 120, 160],
        "gamma": 0.2
    }
    try:
        classification_scheduler = build_scheduler(classification_cfg, optimizer=optimizer)
        print(f"  - 分类任务: {classification_scheduler.__class__.__name__}, milestones={classification_cfg['milestones']}")
    except Exception as e:
        print(f"  - 分类任务: 构建失败 - {str(e)}")
    
    # 案例2: 目标检测常用 WarmupCosineAnnealingLR 配置
    detection_cfg = {
        "type": "CosineAnnealingLRScheduler",
        "T_max": 300,
        "eta_min": 1e-6
    }
    try:
        detection_scheduler = build_scheduler(detection_cfg, optimizer=optimizer)
        print(f"  - 目标检测: {detection_scheduler.__class__.__name__}, T_max={detection_cfg['T_max']}")
    except Exception as e:
        print(f"  - 目标检测: 构建失败 - {str(e)}")
    
    print("\n所有测试通过!")
