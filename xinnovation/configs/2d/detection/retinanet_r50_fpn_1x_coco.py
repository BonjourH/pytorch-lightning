# RetinaNet with ResNet50 backbone and FPN neck
# Training on COCO dataset

model = dict(
    type='DetectionModel2D',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5
    ),
    head=dict(
        type='RetinaNetHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator2D',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(
            type='L1Loss',
            loss_weight=1.0
        )
    )
)

# Dataset settings
data = dict(
    train=dict(
        type='COCODataset',
        ann_file='data/coco/annotations/instances_train2017.json',
        img_prefix='data/coco/train2017',
        transforms=[
            dict(type='Resize', size=(800, 1333)),
            dict(type='RandomFlip', prob=0.5),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ]
    ),
    val=dict(
        type='COCODataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017',
        transforms=[
            dict(type='Resize', size=(800, 1333)),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        ]
    ),
    test=dict(
        type='COCODataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017',
        transforms=[
            dict(type='Resize', size=(800, 1333)),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        ]
    )
)

# Training settings
train = dict(
    work_dir='work_dirs/2d/retinanet_r50_fpn_1x_coco',
    epochs=12,
    batch_size=2,
    num_workers=4,
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.0001
    ),
    scheduler=dict(
        type='CosineAnnealingScheduler',
        T_max=12,
        eta_min=0.00001
    ),
    callbacks=[
        dict(
            type='CheckpointCallback',
            save_dir='checkpoints',
            save_freq=1,
            max_keep=5
        )
    ],
    loggers=[
        dict(
            type='TensorBoardLogger',
            log_dir='logs'
        )
    ],
    strategy=dict(
        type='DDP',
        find_unused_parameters=True
    )
)

# Test settings
test = dict(
    work_dir='work_dirs/2d/retinanet_r50_fpn_1x_coco',
    batch_size=1,
    num_workers=4,
    show=False,
    show_dir='results'
)

# Evaluation settings
evaluation = dict(
    metrics=[
        dict(
            type='MeanAP',
            iou_threshold=0.5,
            num_classes=80
        )
    ],
    analyzers=[
        dict(
            type='ConfusionMatrix',
            num_classes=80
        )
    ],
    visualizers=[
        dict(
            type='GradCAM',
            target_layer='backbone.layer4.2.conv3'
        )
    ]
) 