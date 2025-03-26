# PointPillars on nuScenes dataset
# 3D detection with point cloud input

model = dict(
    type='DetectionModel3D',
    backbone=dict(
        type='PointPillarsBackbone',
        in_channels=64,
        out_channels=[64, 64, 64],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        pillar_size=[0.2, 0.2, 8],
        voxel_size=[0.2, 0.2, 8],
        point_cloud_range=[-50, -50, -5, 50, 50, 3],
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        with_cp=False
    ),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 64, 64],
        out_channels=[128, 128, 128],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01)
    ),
    head=dict(
        type='Anchor3DHead',
        num_classes=10,
        in_channels=384,
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1
            ),
            allowed_border=0,
            pos_weight=-1,
            debug=False
        ),
        test_cfg=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.01,
            score_thr=0.1,
            min_bbox_size=0,
            max_per_img=100
        ),
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[[-50, -50, -1.8, 50, 50, -1.8]],
            sizes=[[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [0.8, 3.9, 2.0]],
            rotations=[0, 1.57],
            matched_threshold=0.5,
            unmatched_threshold=0.35,
            class_name=['car', 'truck', 'construction_vehicle']
        ),
        bbox_coder=dict(
            type='DeltaXYZWLHRBBoxCoder',
            target_means=[.0, .0, .0, .0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(
            type='SmoothL1Loss',
            beta=1.0,
            loss_weight=1.0
        ),
        loss_dir=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.2
        )
    )
)

# Dataset settings
data = dict(
    train=dict(
        type='NuScenesDataset',
        data_root='data/nuscenes',
        ann_file='data/nuscenes/nuscenes_infos_train.pkl',
        pipeline=[
            dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
            dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
            dict(
                type='ObjectNoise',
                num_try=100,
                translation_std=[1.0, 1.0, 0.5],
                global_rot_range=[0.0, 0.0],
                rot_range=[-0.78539816, 0.78539816]
            ),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.78539816, 0.78539816],
                scale_ratio_range=[0.95, 1.05]
            ),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='PointShuffle'),
            dict(type='DefaultFormatBundle3D', class_names=class_names),
            dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=class_names,
        test_mode=False,
        use_valid_flag=True
    ),
    val=dict(
        type='NuScenesDataset',
        data_root='data/nuscenes',
        ann_file='data/nuscenes/nuscenes_infos_val.pkl',
        pipeline=[
            dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1., 1.],
                        translation_std=[0, 0, 0]
                    ),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=point_cloud_range
                    ),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=class_names,
                        with_label=False
                    ),
                    dict(type='Collect3D', keys=['points'])
                ]
            )
        ],
        classes=class_names,
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True
    ),
    test=dict(
        type='NuScenesDataset',
        data_root='data/nuscenes',
        ann_file='data/nuscenes/nuscenes_infos_val.pkl',
        pipeline=[
            dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1., 1.],
                        translation_std=[0, 0, 0]
                    ),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=point_cloud_range
                    ),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=class_names,
                        with_label=False
                    ),
                    dict(type='Collect3D', keys=['points'])
                ]
            )
        ],
        classes=class_names,
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True
    )
)

# Training settings
train = dict(
    work_dir='work_dirs/3d/pointpillars_nuscenes',
    epochs=20,
    batch_size=4,
    num_workers=4,
    optimizer=dict(
        type='AdamW',
        lr=0.001,
        weight_decay=0.01
    ),
    scheduler=dict(
        type='CosineAnnealingScheduler',
        T_max=20,
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
    work_dir='work_dirs/3d/pointpillars_nuscenes',
    batch_size=1,
    num_workers=4,
    show=False,
    show_dir='results'
)

# Evaluation settings
evaluation = dict(
    metrics=[
        dict(
            type='NuScenesMetric',
            result_name='pts_bbox'
        )
    ],
    analyzers=[
        dict(
            type='ConfusionMatrix',
            num_classes=10
        )
    ],
    visualizers=[
        dict(
            type='Det3DLocalVisualizer',
            vis_backends=[dict(type='LocalVisBackend')],
            save_dir='results'
        )
    ]
) 