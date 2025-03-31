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

from xinnovation.src.core import (SourceCameraId, CameraType, CameraParamIndex, EgoStateIndex, TrajParamIndex)

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
        layer_loss_weights=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ),
    detector=dict(
        type="Sparse4DDetector",
        anchor_generator=dict(
            type="Anchor3DGenerator",
            scales=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
        feature_extractors=list(
            dict(
                type="FPNImageFeatureExtractor",
                name="front_stereo_camera",
                camera_group=[SourceCameraId.FRONT_LEFT_CAMERA, SourceCameraId.FRONT_RIGHT_CAMERA],
                backbone="repvgg_a1",
                output_chanels=256,
                fpn_channels=[256, 512, 1024],
                fpn_downsample_scales=[4, 8, 16],
                use_pretrained=True
            ),
            dict(
                type="FPNImageFeatureExtractor",
                name="short_focal_length_camera",
                camera_group=[SourceCameraId.FRONT_CENTER_CAMERA, SourceCameraId.SIDE_LEFT_CAMERA, SourceCameraId.SIDE_RIGHT_CAMERA],
                backbone="repvgg_a1",
                output_chanels=256,
                fpn_channels=[1024],
                fpn_downsample_scales=[16],
                use_pretrained=True
            ),
            dict(
                type="FPNImageFeatureExtractor",
                name="rear_camera",
                camera_group=[SourceCameraId.REAR_LEFT_CAMERA, SourceCameraId.REAR_RIGHT_CAMERA],
                backbone="repvgg_a1",
                output_chanels=256,
                fpn_channels=[256, 512, 1024],
                fpn_downsample_scales=[4, 8, 16],
                use_pretrained=True
            )
        ),
        # mts(multiview_temporal_spatial) feature sampler and aggregator
        mts_feature_sampler=dict(
            type="MultiviewTemporalSpatialFeatureSampler",
            feature_extractors=["front_stereo_camera", "short_focal_length_camera", "rear_camera"],
            temporal_fusion_strategy="average"
        ),
        mts_feature_aggregator=dict(
            type="MultiviewTemporalSpatialFeatureAggregator",
            feature_extractors=["front_stereo_camera", "short_focal_length_camera", "rear_camera"],
            temporal_fusion_strategy="average"
        ),
    )
)
# ============================== 3. Data Config ==============================


# ============================== 4. Trainer Config ==============================


# ============================== 5. Evaluation Config ==============================