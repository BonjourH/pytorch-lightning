# 顶层注册表（版本控制）
LIGHTNING = Registry("lightning", version="1.0")

# ==========================================
#              核心模型体系
# ==========================================
MODEL = LIGHTNING.create_child("model")

# ---- 模型架构 ----
ARCHITECTURES = MODEL.create_child("arch")  # 完整算法架构
ARCHITECTURES.create_child("detector")     # 检测算法 (YOLO, CenterPoint)
ARCHITECTURES.create_child("classifier")   # 分类算法 (ResNet, ViT)
ARCHITECTURES.create_child("segmentor")    # 分割算法 (PointRCNN, MaskRCNN)

# ---- 原子组件 ----
COMPONENTS = MODEL.create_child("components")
COMPONENTS.create_child("backbone")    # 特征提取 (PointNet++, VoxelNet)
COMPONENTS.create_child("neck")        # 特征融合 (FPN, PointFPN)
COMPONENTS.create_child("head")        # 任务头 (AnchorHead, CenterHead)
COMPONENTS.create_child("attention")   # 注意力模块 (TransformerBlock)
COMPONENTS.create_child("embedding")   # 嵌入层 (Sinusoidal, Learned)

# ---- 优化体系 ----
OPTIMIZATION = MODEL.create_child("optim")
OPTIMIZATION.create_child("loss")         # 损失函数 (FocalLoss, L1)
OPTIMIZATION.create_child("optimizer")    # 优化器 (AdamW, Lion)
OPTIMIZATION.create_child("scheduler")    # 学习策略 (Cosine, OneCycle)

# ==========================================
#              数据管道体系
# ==========================================
DATA = LIGHTNING.create_child("data")

# ---- 数据源 ----
SOURCES = DATA.create_child("source")
SOURCES.create_child("kitti")        # KITTI数据集
SOURCES.create_child("nuscenes")     # nuScenes数据集

# ---- 处理流程 ----
PIPELINES = DATA.create_child("pipeline")
PIPELINES.create_child("transform")  # 数据增强
PIPELINES.create_child("sampler")    # 数据采样
PIPELINES.create_child("loader")     # 加载策略

# ==========================================
#              训练生态系统
# ==========================================
TRAINER = LIGHTNING.create_child("train")

# ---- 训练核心 ----
CORE = TRAINER.create_child("core")
CORE.create_child("strategy")    # 训练策略 (DDP, FSDP)
CORE.create_child("precision")  # 精度模式 (16/32/mixed)

# ---- 训练支持 ----
SUPPORT = TRAINER.create_child("support")
SUPPORT.create_child("callback")    # 训练回调
SUPPORT.create_child("logger")      # 实验日志
SUPPORT.create_child("profiler")    # 性能分析

# ==========================================
#              生产化体系
# ==========================================
DEPLOY = LIGHTNING.create_child("deploy")

# ---- 转换优化 ----
CONVERSION = DEPLOY.create_child("convert")
CONVERSION.create_child("onnx")      # ONNX转换
CONVERSION.create_child("tensorrt")  # TensorRT优化

# ---- 加速推理 ----
INFERENCE = DEPLOY.create_child("infer")
INFERENCE.create_child("quantize")   # 量化 (INT8/QAT)
INFERENCE.create_child("prune")      # 剪枝 (L1/L2)
INFERENCE.create_child("compile")    # 编译优化 (TVM)

# ==========================================
#              评估诊断体系
# ==========================================
EVAL = LIGHTNING.create_child("eval")

# ---- 量化评估 ----
METRICS = EVAL.create_child("metric")
METRICS.create_child("detection")   # mAP, NDS
METRICS.create_child("segmentation")# IoU, mIoU

# ---- 可视化 ----
VIS = EVAL.create_child("vis")
VIS.create_child("feature")   # 特征图可视化
VIS.create_child("gradcam")   # 类激活图