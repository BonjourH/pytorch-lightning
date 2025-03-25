# registry.py
from typing import Dict, Any, Callable, Optional, Type, Union, List
import inspect
from collections import defaultdict

class Registry:
    """模块化组件注册中心（支持多级嵌套结构）
    
    特性：
    1. 分层注册管理：支持树状结构组件组织
    2. 动态构建系统：通过配置字典实例化组件
    3. 组件分组管理：支持按功能分组
    4. 循环依赖检测：构建时自动检查循环依赖
    """
    
    def __init__(self, name: str):
        """初始化注册表
        
        Args:
            name (str): 注册表名称，用于调试和显示
        """
        self._name = name
        self._module_dict: Dict[str, Type] = {}
        self._children: Dict[str, 'Registry'] = {}
        self._group_dict: Dict[str, List[str]] = defaultdict(list)
        self._building_stack = []  # 用于检测循环依赖

    def __repr__(self):
        return f"Registry(name={self._name}, items={len(self)} children={len(self._children)})"

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return key in self._module_dict

    def __getitem__(self, key):
        return self.get(key)

    def create_child(self, name: str) -> 'Registry':
        """创建子注册表"""
        if name in self._children:
            return self._children[name]
            
        child = Registry(f"{self._name}.{name}")
        self._children[name] = child
        return child

    def register_module(
        self, 
        name: Optional[str] = None, 
        module: Optional[Type] = None, 
        force: bool = False,
        group: Optional[str] = None
    ) -> Callable:
        """注册模块的装饰器/函数"""
        if module is None:
            def _register(cls):
                cls_name = name or cls.__name__
                self._register_module(cls_name, cls, force, group)
                return cls
            return _register
        
        module_name = name or module.__name__
        self._register_module(module_name, module, force, group)
        return module

    def _register_module(self, name: str, module: Type, force: bool, group: Optional[str]):
        """实际注册实现"""
        if not force and name in self._module_dict:
            raise KeyError(f"{name} already exists in {self._name}")
        self._module_dict[name] = module
        if group:
            self._group_dict[group].append(name)

    def get(self, key: str) -> Type:
        """获取注册的类/函数"""
        if key not in self._module_dict:
            available = list(self._module_dict.keys())
            raise KeyError(f"{key} not found in {self._name}. Available: {available}")
        return self._module_dict[key]

    def get_group(self, group_name: str) -> List[str]:
        """获取指定分组的所有组件名称
        
        Args:
            group_name (str): 分组名称
            
        Returns:
            List[str]: 该分组下的所有组件名称列表
        """
        return self._group_dict[group_name]

    def build(self, cfg: Union[Dict, Type], *args, **kwargs) -> Any:
        """根据配置构建实例（核心方法）"""
        if cfg is None:
            return None
            
        # 直接处理类或实例
        if not isinstance(cfg, dict):
            return cfg(*args, **kwargs) if inspect.isclass(cfg) else cfg
            
        cfg = cfg.copy()
        obj_type = cfg.pop("type")
        
        # 循环依赖检测
        if obj_type in self._building_stack:
            raise RuntimeError(f"Cyclic dependency detected: {self._building_stack + [obj_type]}")
        self._building_stack.append(obj_type)
        
        try:
            obj_cls = self.get(obj_type)
            # 递归构建嵌套配置
            for k, v in cfg.items():
                if isinstance(v, dict) and "type" in v:
                    cfg[k] = self.build(v)
            return obj_cls(*args, **kwargs, **cfg)
        except RuntimeError:
            # 直接重新抛出循环依赖错误
            raise
        except Exception as e:
            # 生成友好的错误提示
            sig = inspect.signature(obj_cls.__init__)
            params = [f"{p.name}={p.default}" for p in sig.parameters.values() 
                     if p.name not in ['self', 'args', 'kwargs']]
            raise TypeError(
                f"Failed to build {obj_type}: {e}\n"
                f"Expected params: {', '.join(params)}"
            )
        finally:
            self._building_stack.pop()

    def get_module_dict(self) -> Dict[str, Type]:
        """获取所有注册的模块字典"""
        return self._module_dict.copy()

    def get_registered_modules(self) -> List[str]:
        """获取所有注册的模块名称列表"""
        return list(self._module_dict.keys())

# LIGHTNING (顶层)
# ├── MODEL
# │   ├── BACKBONES          # 骨干网络 (ResNet, SwinTransformer, PointNet++)
# │   ├── NECKS              # 特征融合 (FPN, BiFPN, ASFF)
# │   ├── HEADS              # 任务头 (DetectionHead, ClassificationHead)
# │   ├── LOSSES             # 损失函数 (FocalLoss, SmoothL1, DiceLoss)
# │   ├── ATTENTION          # 注意力机制 (SelfAttention, CBAM, TransformerBlock)
# │   ├── NORM_LAYERS        # 归一化层 (BatchNorm, LayerNorm, GroupNorm)
# │   └── POS_ENCODING       # 位置编码 (SineEncoding, LearnedEncoding)
# ├── DATA
# │   ├── DATASETS           # 数据集 (ImageNet, COCO, KITTI, nuScenes)
# │   ├── TRANSFORMS         # 数据增强 (RandomFlip, Normalize, ColorJitter)
# │   ├── SAMPLERS           # 数据采样 (BalancedSampler, WeightedRandomSampler)
# │   └── LOADERS            # 数据加载策略 (Dataloader, PrefetchLoader)
# ├── TRAINER
# │   ├── OPTIMIZERS         # 优化器 (SGD, AdamW, Lion)
# │   ├── SCHEDULERS         # 学习率策略 (StepLR, CosineAnnealingLR)
# │   ├── CALLBACKS          # 训练回调 (Checkpoint, EarlyStopping, GradientClipping)
# │   ├── LOGGERS            # 训练日志 (TensorBoard, WandB, CSVLogger)
# │   └── STRATEGIES         # 并行训练策略 (DDP, DeepSpeed, FSDP)
# ├── EVALUATION
# │   ├── METRICS            # 评估指标 (mAP, IoU, F1-score)
# │   ├── ANALYZERS          # 结果分析工具 (ConfusionMatrix, PR-Curve)
# │   └── VISUALIZERS        # 结果可视化 (GradCAM, FeatureMapViewer)
# ├── DEPLOY
# │   ├── CONVERTERS         # 格式转换 (ONNX, TorchScript, CoreML)
# │   ├── QUANTIZERS         # 量化工具 (INT8量化, PTQ, QAT)
# │   ├── PRUNERS            # 模型剪枝 (L1, L2, LotteryTicket)
# │   ├── COMPRESSORS        # 模型压缩 (Knowledge Distillation, Huffman Coding)
# │   └── RUNTIME            # 运行时优化 (TensorRT, OpenVINO, TVM)
# └── MULTIMODAL
#     ├── FUSION             # 特征融合策略 (Late Fusion, Attention Fusion)
#     ├── ALIGNMENT          # 模态对齐 (Cross-Modal Contrastive Learning)
#     └── EMBEDDING          # 跨模态嵌入 (CLIP, ALIGN, ImageBind)

# ==========================================
#              LIGHTNING 注册表
# ==========================================

# 顶层注册表
LIGHTNING = Registry("lightning")

# ==========================================
#                模型组件
# ==========================================
MODEL = LIGHTNING.create_child("model")

# 主要模块
BACKBONES = MODEL.create_child("backbone")  # 骨干网络 (ResNet, SwinTransformer, PointNet++)
NECKS = MODEL.create_child("neck")          # 特征融合 (FPN, BiFPN, ASFF)
HEADS = MODEL.create_child("head")          # 任务头 (DetectionHead, ClassificationHead)
LOSSES = MODEL.create_child("loss")         # 损失函数 (FocalLoss, SmoothL1, DiceLoss)

# 其他常用模块
ATTENTION = MODEL.create_child("attention")     # 注意力机制 (SelfAttention, CBAM, TransformerBlock)
NORM_LAYERS = MODEL.create_child("norm_layer")  # 归一化层 (BatchNorm, LayerNorm, GroupNorm)
POS_ENCODING = MODEL.create_child("pos_encoding")  # 位置编码 (SineEncoding, LearnedEncoding)

# ==========================================
#                数据组件
# ==========================================
DATA = LIGHTNING.create_child("data")

DATASETS = DATA.create_child("dataset")     # 数据集 (ImageNet, COCO, KITTI, nuScenes)
TRANSFORMS = DATA.create_child("transform") # 数据增强 (RandomFlip, Normalize, ColorJitter)
SAMPLERS = DATA.create_child("sampler")     # 采样策略 (BalancedSampler, WeightedRandomSampler)
LOADERS = DATA.create_child("loader")       # 数据加载策略 (Dataloader, PrefetchLoader)

# ==========================================
#               训练组件
# ==========================================
TRAINER = LIGHTNING.create_child("trainer")

OPTIMIZERS = TRAINER.create_child("optimizer")  # 优化器 (SGD, AdamW, Lion)
SCHEDULERS = TRAINER.create_child("scheduler")  # 学习率策略 (StepLR, CosineAnnealingLR)
CALLBACKS = TRAINER.create_child("callback")    # 训练回调 (Checkpoint, EarlyStopping, GradientClipping)
LOGGERS = TRAINER.create_child("logger")        # 训练日志 (TensorBoard, WandB, CSVLogger)
STRATEGIES = TRAINER.create_child("strategy")   # 并行训练策略 (DDP, DeepSpeed, FSDP)

# ==========================================
#               评估体系
# ==========================================
EVALUATION = LIGHTNING.create_child("evaluation")

METRICS = EVALUATION.create_child("metric")       # 评估指标 (mAP, IoU, F1-score)
ANALYZERS = EVALUATION.create_child("analyzer")   # 结果分析工具 (ConfusionMatrix, PR-Curve)
VISUALIZERS = EVALUATION.create_child("visualizer") # 结果可视化 (GradCAM, FeatureMapViewer)

# ==========================================
#               部署优化
# ==========================================
DEPLOY = LIGHTNING.create_child("deploy")

CONVERTERS = DEPLOY.create_child("converter")  # 格式转换 (ONNX, TorchScript, CoreML)
QUANTIZERS = DEPLOY.create_child("quantizer")  # 量化工具 (INT8量化, PTQ, QAT)
PRUNERS = DEPLOY.create_child("pruner")        # 模型剪枝 (L1, L2, LotteryTicket)
COMPRESSORS = DEPLOY.create_child("compressor") # 模型压缩 (Knowledge Distillation, Huffman Coding)
RUNTIME = DEPLOY.create_child("runtime")       # 运行时优化 (TensorRT, OpenVINO, TVM)

# ==========================================
#              多模态处理
# ==========================================
MULTIMODAL = LIGHTNING.create_child("multimodal")

FUSION = MULTIMODAL.create_child("fusion")          # 特征融合策略 (Late Fusion, Attention Fusion)
ALIGNMENT = MULTIMODAL.create_child("alignment")    # 模态对齐 (Cross-Modal Contrastive Learning)
EMBEDDING = MULTIMODAL.create_child("embedding")    # 跨模态嵌入 (CLIP, ALIGN, ImageBind)


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from lightning.pytorch import LightningModule
    
    # 测试1: 注册和构建基础组件
    @BACKBONES.register_module()
    class ResNet(nn.Module):
        def __init__(self, depth=50, pretrained=None):
            super().__init__()
            self.depth = depth
            self.pretrained = pretrained
            
        def forward(self, x):
            return x
    
    # 测试2: 注册带分组的组件
    @HEADS.register_module(group="detection")
    class DetectionHead(nn.Module):
        def __init__(self, in_channels, num_classes):
            super().__init__()
            self.in_channels = in_channels
            self.num_classes = num_classes
            
        def forward(self, x):
            return x
    
    # 测试3: 注册多模态组件
    @FUSION.register_module()
    class FeatureFusion(nn.Module):
        def __init__(self, fusion_type="concat"):
            super().__init__()
            self.fusion_type = fusion_type
            
        def forward(self, x1, x2):
            return x1 + x2
    
    # 测试4: 注册部署相关组件
    @CONVERTERS.register_module()
    class ONNXConverter:
        def __init__(self, opset_version=11):
            self.opset_version = opset_version
            
        def convert(self, model, dummy_input):
            return model
    
    # 测试5: 注册评估组件
    @METRICS.register_module()
    class MeanAP:
        def __init__(self, iou_threshold=0.5):
            self.iou_threshold = iou_threshold
            
        def compute(self, pred, target):
            return 0.95
    
    # 测试用例执行
    print("\n=== 测试1: 基础组件注册和构建 ===")
    backbone_cfg = {"type": "ResNet", "depth": 101}
    backbone = BACKBONES.build(backbone_cfg)
    print(f"构建的backbone: {backbone}")
    
    print("\n=== 测试2: 分组功能 ===")
    detection_heads = HEADS.get_group("detection")
    print(f"检测头组件: {detection_heads}")
    
    print("\n=== 测试3: 多模态组件 ===")
    fusion_cfg = {"type": "FeatureFusion", "fusion_type": "concat"}
    fusion = FUSION.build(fusion_cfg)
    print(f"构建的特征融合模块: {fusion}")
    
    print("\n=== 测试4: 部署组件 ===")
    converter_cfg = {"type": "ONNXConverter", "opset_version": 12}
    converter = CONVERTERS.build(converter_cfg)
    print(f"构建的转换器: {converter}")
    
    print("\n=== 测试5: 评估组件 ===")
    metric_cfg = {"type": "MeanAP", "iou_threshold": 0.5}
    metric = METRICS.build(metric_cfg)
    print(f"构建的评估指标: {metric}")
    
    print("\n=== 测试6: 循环依赖检测 ===")
    try:
        cyclic_cfg = {
            "type": "ResNet",
            "backbone": {
                "type": "ResNet",
                "backbone": {
                    "type": "ResNet"
                }
            }
        }
        BACKBONES.build(cyclic_cfg)
    except RuntimeError as e:
        print(f"成功捕获循环依赖: {e}")
    except Exception as e:
        print(f"发生其他错误: {e}")
    
    print("\n=== 测试7: 注册表信息 ===")
    print(f"MODEL注册表: {MODEL}")
    print(f"可用组件: {MODEL.get_registered_modules()}")
    
    print("\n=== 测试8: 嵌套配置构建 ===")
    model_cfg = {
        "type": "DetectionModel",
        "backbone": {
            "type": "ResNet",
            "depth": 50
        },
        "head": {
            "type": "DetectionHead",
            "in_channels": 2048,
            "num_classes": 80
        }
    }
    print(f"嵌套配置示例: {model_cfg}")
    