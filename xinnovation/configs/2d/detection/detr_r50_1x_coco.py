# DETR with ResNet50 backbone
# Training on COCO dataset

model = dict(
    type='DetectionModel2D',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),  # DETR only uses the last stage
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[2048],
        out_channels=256,
        kernel_size=1
    ),
    head=dict(
        type='DETRHead',
        num_classes=80,
        num_queries=100,
        in_channels=256,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        num_encoder_stages=1,
        num_decoder_stages=1,
        pre_norm=False,
        with_box_refine=False,
        as_two_stage=False,
        transformer=dict(
            type='DetrTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=1,
                        num_points=4
                    ),
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')
                )
            ),
            decoder=dict(
                type='DetrTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1
                        ),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=1,
                            num_points=4
                        )
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                   'ffn', 'norm')
                )
            )
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True
        ),
        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0
        ),
        loss_bbox=dict(
            type='L1Loss',
            loss_weight=5.0
        ),
        loss_iou=dict(
            type='GIoULoss',
            loss_weight=2.0
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
    work_dir='work_dirs/2d/detr_r50_1x_coco',
    epochs=50,  # DETR typically needs more epochs
    batch_size=2,
    num_workers=4,
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.0001,
        paramwise_cfg=dict(
            custom_keys={
                'backbone': dict(lr_mult=0.1),
                'sampling_offsets': dict(lr_mult=0.1),
                'reference_points': dict(lr_mult=0.1)
            }
        )
    ),
    scheduler=dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=1000
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
    work_dir='work_dirs/2d/detr_r50_1x_coco',
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