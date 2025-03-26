import torch
from torch.optim import AdamW, Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, LBFGS, NAdam, RAdam, SparseAdam
from typing import Optional, Dict, Any, Union
from xinnovation.src.core.registry import OPTIMIZERS

__all__ = [
    'AdamWOptimizer',
    'AdamOptimizer',
    'SGDOptimizer',
    'RMSpropOptimizer',
    'AdagradOptimizer',
    'AdadeltaOptimizer',
    'AdamaxOptimizer',
    'LBFGSOptimizer',
    'NAdamOptimizer',
    'RAdamOptimizer',
    'SparseAdamOptimizer',
]

@OPTIMIZERS.register_module(force=True)
class AdamWOptimizer:
    """AdamW optimizer wrapper with weight decay.

    Args:
        lr (float): Learning rate
        betas (tuple): Beta parameters
        eps (float): Epsilon parameter
        weight_decay (float): Weight decay
        amsgrad (bool): Whether to use AMSGrad variant
    """

    def __init__(
        self,
        params: Optional[torch.nn.Parameter] = None,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=False,
        **kwargs
    ):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.optimizer = AdamW(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad
        )


@OPTIMIZERS.register_module(force=True)
class AdamOptimizer:
    """Adam optimizer wrapper.

    Args:
        lr (float): 学习率
        betas (tuple): 衰减系数
        eps (float): 稳定因子
        weight_decay (float): 权重衰减
        amsgrad (bool): 是否使用AMSGrad
    """
    
    def __init__(
        self,
        params: Optional[torch.nn.Parameter] = None,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=False,
        **kwargs
    ):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.optimizer = Adam(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad
        )


@OPTIMIZERS.register_module(force=True)
class SGDOptimizer:
    """SGD optimizer wrapper.

    Args:
        lr (float): 学习率
        momentum (float): 动量
        weight_decay (float): 权重衰减
        dampening (float): 阻尼
        nesterov (bool): 是否使用Nesterov动量
    """
    
    def __init__(
        self,
        params: Optional[torch.nn.Parameter] = None,
        lr=1e-3,
        momentum=0.9,
        weight_decay=0.01,
        dampening=0,
        nesterov=False,
        **kwargs
    ):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov
        self.optimizer = SGD(
            params,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            dampening=self.dampening,
            nesterov=self.nesterov
        )


@OPTIMIZERS.register_module(force=True)
class RMSpropOptimizer:
    """RMSprop optimizer wrapper.

    Args:
        lr (float): 学习率
        momentum (float): 动量
        weight_decay (float): 权重衰减
        dampening (float): 阻尼
        nesterov (bool): 是否使用Nesterov动量
    """
    
    def __init__(
        self,
        params: Optional[torch.nn.Parameter] = None,
        lr=1e-3,
        momentum=0.9,
        weight_decay=0.01,
        dampening=0,
        nesterov=False,
        **kwargs
    ):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov
        self.optimizer = RMSprop(
            params,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            dampening=self.dampening,
            nesterov=self.nesterov
        )


@OPTIMIZERS.register_module(force=True)
class AdagradOptimizer:
    """Adagrad optimizer wrapper.

    Args:
        lr (float): 学习率
        weight_decay (float): 权重衰减
        eps (float): 稳定因子
    """
    
    def __init__(
        self,
        params: Optional[torch.nn.Parameter] = None,
        lr=1e-3,
        weight_decay=0.01,
        eps=1e-8,
        **kwargs
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.optimizer = Adagrad(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=self.eps
        )


@OPTIMIZERS.register_module(force=True)
class AdadeltaOptimizer:
    """Adadelta optimizer wrapper.

    Args:
        lr (float): 学习率
        rho (float): 梯度平方衰减系数
        eps (float): 稳定因子
        weight_decay (float): 权重衰减
    """
    
    def __init__(
        self,
        params: Optional[torch.nn.Parameter] = None,
        lr=1.0,
        rho=0.9,
        eps=1e-6,
        weight_decay=0.0,
        **kwargs
    ):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        self.optimizer = Adadelta(
            params,
            lr=self.lr,
            rho=self.rho,
            eps=self.eps,
            weight_decay=self.weight_decay
        )


@OPTIMIZERS.register_module(force=True)
class AdamaxOptimizer:
    """Adamax optimizer wrapper (Adam的变体之一).

    Args:
        lr (float): 学习率
        betas (tuple): 梯度和梯度平方的衰减系数
        eps (float): 稳定因子
        weight_decay (float): 权重衰减
    """
    
    def __init__(
        self,
        params: Optional[torch.nn.Parameter] = None,
        lr=2e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        **kwargs
    ):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.optimizer = Adamax(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay
        )


@OPTIMIZERS.register_module(force=True)
class LBFGSOptimizer:
    """LBFGS optimizer wrapper.
    
    注意: 这是一个基于内存的优化器，不适合大型模型或大批量训练

    Args:
        lr (float): 学习率
        max_iter (int): 每步更新的最大迭代次数
        max_eval (int): 每步函数评估的最大次数
        tolerance_grad (float): 梯度收敛的容差
        tolerance_change (float): 函数值/参数变化的容差
        history_size (int): 存储的历史数据大小
        line_search_fn (str): 使用的线搜索函数('strong_wolfe'或None)
    """
    
    def __init__(
        self,
        params: Optional[torch.nn.Parameter] = None,
        lr=1.0,
        max_iter=20,
        max_eval=None,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn=None,
        **kwargs
    ):
        self.lr = lr
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change
        self.history_size = history_size
        self.line_search_fn = line_search_fn
        self.optimizer = LBFGS(
            params,
            lr=self.lr,
            max_iter=self.max_iter,
            max_eval=self.max_eval,
            tolerance_grad=self.tolerance_grad,
            tolerance_change=self.tolerance_change,
            history_size=self.history_size,
            line_search_fn=self.line_search_fn
        )


@OPTIMIZERS.register_module(force=True)
class NAdamOptimizer:
    """NAdam optimizer wrapper (Nesterov加速的Adam优化器).

    Args:
        lr (float): 学习率
        betas (tuple): 梯度和梯度平方的衰减系数
        eps (float): 稳定因子
        weight_decay (float): 权重衰减
        momentum_decay (float): 动量衰减
    """
    
    def __init__(
        self,
        params: Optional[torch.nn.Parameter] = None,
        lr=2e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        momentum_decay=4e-3,
        **kwargs
    ):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum_decay = momentum_decay
        self.optimizer = NAdam(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            momentum_decay=self.momentum_decay
        )


@OPTIMIZERS.register_module(force=True)
class RAdamOptimizer:
    """RAdam optimizer wrapper (修正的Adam优化器).

    Args:
        lr (float): 学习率
        betas (tuple): 梯度和梯度平方的衰减系数
        eps (float): 稳定因子
        weight_decay (float): 权重衰减
    """
    
    def __init__(
        self,
        params: Optional[torch.nn.Parameter] = None,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        **kwargs
    ):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.optimizer = RAdam(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay
        )


@OPTIMIZERS.register_module(force=True)
class SparseAdamOptimizer:
    """SparseAdam optimizer wrapper (适用于稀疏梯度的Adam优化器).

    Args:
        lr (float): 学习率
        betas (tuple): 梯度和梯度平方的衰减系数
        eps (float): 稳定因子
    """
    
    def __init__(
        self,
        params: Optional[torch.nn.Parameter] = None,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        **kwargs
    ):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.optimizer = SparseAdam(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps
        )


# ====================================
# 验证用例
# ====================================

if __name__ == "__main__":
    """验证各种优化器的构建过程"""
    
    from xinnovation.src.core.builders import build_optimizer
    # 检查注册表中的优化器
    print("\n===== 注册表状态 =====")
    registered_optimizers = OPTIMIZERS.get_module_dict().keys()
    print(f"当前已注册的优化器: {', '.join(registered_optimizers)}")
    
    # 创建一个简单模型获取参数
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 10)
    )
    
    print("\n===== 优化器构建测试 =====")
    
    # 测试1: 基础优化器构建
    print("\n[测试1] 基础优化器构建:")
    
    optimizers = [
        ("AdamW", {"type": "AdamWOptimizer", "lr": 0.001, "weight_decay": 0.01}),
        ("Adam", {"type": "AdamOptimizer", "lr": 0.002, "amsgrad": True}),
        ("SGD", {"type": "SGDOptimizer", "lr": 0.01, "momentum": 0.9, "nesterov": True}),
        ("RMSprop", {"type": "RMSpropOptimizer", "lr": 0.0005}),
        ("Adagrad", {"type": "AdagradOptimizer", "lr": 0.01}),
    ]
    
    for name, cfg in optimizers:
        try:
            optimizer = build_optimizer(cfg, params=model.parameters())
            print(f"  - {name}: {optimizer.__class__.__name__}, lr={optimizer.param_groups[0]['lr']}")
        except Exception as e:
            print(f"  - {name}: 构建失败 - {str(e)}")
    
    # 测试2: 高级优化器构建
    print("\n[测试2] 高级优化器构建:")
    
    advanced_optimizers = [
        ("Adadelta", {"type": "AdadeltaOptimizer", "rho": 0.95}),
        ("Adamax", {"type": "AdamaxOptimizer", "lr": 0.003}),
        ("NAdam", {"type": "NAdamOptimizer", "lr": 0.002}),
        ("RAdam", {"type": "RAdamOptimizer", "lr": 0.001, "weight_decay": 0.001}),
    ]
    
    for name, cfg in advanced_optimizers:
        try:
            optimizer = build_optimizer(cfg, params=model.parameters())
            print(f"  - {name}: {optimizer.__class__.__name__}, lr={optimizer.param_groups[0]['lr']}")
        except Exception as e:
            print(f"  - {name}: 构建失败 - {str(e)}")
    
    # 测试3: 特殊优化器构建
    print("\n[测试3] 特殊优化器构建:")
    
    # LBFGS优化器 (通常用于小批量数据)
    lbfgs_cfg = {
        "type": "LBFGSOptimizer", 
        "lr": 0.1,
        "max_iter": 5,
        "history_size": 10
    }
    try:
        lbfgs_optimizer = build_optimizer(lbfgs_cfg, params=model.parameters())
        print(f"  - LBFGS: {lbfgs_optimizer.__class__.__name__}, lr={lbfgs_optimizer.param_groups[0]['lr']}")
    except Exception as e:
        print(f"  - LBFGS: 构建失败 - {str(e)}")
    
    # 测试4: 参数分组
    print("\n[测试4] 参数分组测试:")
    
    # 为不同层设置不同的学习率
    param_groups = [
        {'params': model[0].parameters(), 'lr': 0.01, 'weight_decay': 0.001},  # 第一层
        {'params': model[2].parameters(), 'lr': 0.001}                         # 第二层
    ]
    
    grouped_cfg = {"type": "AdamWOptimizer", "weight_decay": 0.01}
    try:
        grouped_optimizer = build_optimizer(grouped_cfg, params=param_groups)
        
        print("  - 参数分组:")
        for i, group in enumerate(grouped_optimizer.param_groups):
            print(f"    组 {i+1}: lr={group['lr']}, weight_decay={group.get('weight_decay', 'N/A')}")
    except Exception as e:
        print(f"  - 参数分组: 构建失败 - {str(e)}")
    
    # 测试5: 配置案例
    print("\n[测试5] 常用配置案例:")
    
    # 案例1: YOLO训练常用配置
    yolo_cfg = {
        "type": "SGDOptimizer",
        "lr": 0.01,
        "momentum": 0.937,
        "nesterov": True,
        "weight_decay": 0.0005
    }
    try:
        yolo_optimizer = build_optimizer(yolo_cfg, params=model.parameters())
        print(f"  - YOLO训练: {yolo_optimizer.__class__.__name__}, lr={yolo_optimizer.param_groups[0]['lr']}, momentum={yolo_optimizer.param_groups[0]['momentum']}")
    except Exception as e:
        print(f"  - YOLO训练: 构建失败 - {str(e)}")
    
    # 案例2: Transformer训练常用配置
    transformer_cfg = {
        "type": "AdamWOptimizer",
        "lr": 0.0001,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01
    }
    try:
        transformer_optimizer = build_optimizer(transformer_cfg, params=model.parameters())
        print(f"  - Transformer训练: {transformer_optimizer.__class__.__name__}, lr={transformer_optimizer.param_groups[0]['lr']}, weight_decay={transformer_optimizer.param_groups[0]['weight_decay']}")
    except Exception as e:
        print(f"  - Transformer训练: 构建失败 - {str(e)}")
    
    print("\n所有测试通过!")


