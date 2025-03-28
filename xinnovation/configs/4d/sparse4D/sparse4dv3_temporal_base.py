# ============================== 1. Base Config ==============================


# ============================== 2. Model Config ==============================
# ├── LIGHTNING_MODULE     # 模型训练器 (Sparse4DModule, PointNet2Module)
# │   ├── OPTIMIZERS         # 优化器 (SGD, AdamW, Lion)
# │   ├── SCHEDULERS         # 学习率策略 (StepLR, CosineAnnealingLR)
# │   ├── LOSSES             # 损失函数 (FocalLoss, SmoothL1, DiceLoss)
# │   ├── DETECTORS          # 检测器 (YOLO, SSD, FasterRCNN)
# │   │   ├── BACKBONES          # 骨干网络 (ResNet, SwinTransformer, PointNet++)
# │   │   ├── NECKS              # 特征融合 (FPN, BiFPN, ASFF)
# │   │   ├── HEADS              # 任务头 (DetectionHead, ClassificationHead)
# │   │   ├── ATTENTION          # 注意力机制 (SelfAttention, CBAM, TransformerBlock)
# │   │   ├── NORM_LAYERS        # 归一化层 (BatchNorm, LayerNorm, GroupNorm)
# │   │   └── POS_ENCODING       # 位置编码 (SineEncoding, LearnedEncoding)

lightning_module = dict(
    type="Sparse4DModule",
    scheduler=dict(
        type="StepLRScheduler",
        step_size=10,
        gamma=0.1
    ),
    optimizer=dict(
        type="AdamWOptimizer",
        lr=0.001,
        weight_decay=0.0001
    ),
    loss=dict(
        type="Sparse4DLossWithDAC",
        layer_loss_weights=[0.5, 0.5],
        loss_cls=dict(
            type="FocalLoss",
            alpha=0.25,
            gamma=2.0,
            reduction="mean"
        ),
        loss_reg=dict(
            type="SmoothL1Loss",
            beta=1.0
        )
    ),
    detector=dict(
        type="Sparse4DDetector",
        backbone=dict(
            type="ResNet",
            depth=50
        ),
        neck=dict(
            type="FPN",
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs="on_output"
        ),
        head=dict(
            type="Sparse4DHead",
            in_channels=256,
            out_channels=256,
            num_classes=10
        )
    )
)
# ============================== 3. Data Config ==============================


# ============================== 4. Trainer Config ==============================


# ============================== 5. Evaluation Config ==============================