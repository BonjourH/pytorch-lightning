from typing import List, Dict
from xinnovation.src.core import (SourceCameraId, CameraType, CameraParamIndex, EgoStateIndex, TrajParamIndex)
from xinnovation.examples.detector4D.sparse4d_dataset import CameraGroupConfig


# ============================== 1. Base Config ==============================
query_dim = 256
seq_length = 5
dropout = 0.1
num_groups = 8
num_decoder = 6
num_classes = TrajParamIndex.END_OF_INDEX - TrajParamIndex.HAS_OBJECT
use_temp_attention = False
with_quality_estimation = False
wandb_project_name = "sparse4d_v1"
exp_name = "sparse4dv3_temporal_base"
work_dir = "/home/xinjian/Code/pytorch-lightning"
save_dir = f"{work_dir}/xinnovation_checkpoints"
checkpoint_dir = f"{work_dir}/xinnovation_checkpoints"
epochs = 10
accumulate_grad_batches = 1
resume = False
shuffle = False
batch_size = 1
devices = [0]
xrel_range = [-80.0, 250.0]
yrel_range = [-10.0, 10.0]
# devices = [0]

# ============================== 2. Trainer Config ==============================
lightning_trainer = dict(
    type="LightningTrainer",
    # Training loop parameters
    max_epochs=epochs,
    min_epochs=None,
    max_steps=-1,
    min_steps=None,
    max_time=None,
    limit_train_batches=1, # 1 for one epoch, 1.0 for all epochs
    limit_val_batches=1,  
    limit_test_batches=1,
    limit_predict_batches=1,
    overfit_batches=0.0,
    val_check_interval=1.0,
    check_val_every_n_epoch=1,
    num_sanity_val_steps=0, # 0 remove sanity check
    log_every_n_steps=50,
    enable_progress_bar=True,
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    deterministic=True,
    # Acceleration parameters
    accelerator="gpu",
    devices=devices,
    num_nodes=1,
    precision="32",
    strategy=dict(
        type="LightningDDPStrategy",
        find_unused_parameters=True
    ),
    accumulate_grad_batches=accumulate_grad_batches,
    sync_batchnorm=False,
    use_distributed_sampler=True,
    benchmark=False,
    plugins=None,
    # Checkpoint and logger parameters
    enable_checkpointing=True,
    callbacks=[
        dict(type="CheckpointCallback", dirpath=save_dir, filename='epoch-{epoch:02d}', save_top_k=2, save_last=True, monitor="train/loss_epoch"),
        dict(type="LearningRateMonitorCallback", logging_interval="step"),
        dict(type="FilteredProgressBarCallback", metrics_to_display=["train/loss_epoch", "val/loss"]),
        # dict(type="EarlyStoppingCallback", monitor="train/loss_epoch", mode="min", patience=10, min_delta=0.0001),
    ],
    logger=[
        # dict(type="LightningWandbLogger", project=wandb_project_name, name=exp_name, save_dir=save_dir, keys_to_log=[], use_optional_metrics=False),
        dict(type="LightningTensorBoardLogger", name=exp_name, save_dir=save_dir, default_hp_metric=False),
        # dict(type="LightningCSVLogger", save_dir=save_dir),
    ],
    default_root_dir=save_dir,
    enable_model_summary=True,
    # Profiling parameters
    profiler=None,
    detect_anomaly=True,
    # Other parameters
    fast_dev_run=False,
    reload_dataloaders_every_n_epochs=0,
)

# ============================== 3. Data Config ==============================
lightning_data_module = dict(
    type="Sparse4DDataModule",
    train_list=f"{work_dir}/train_clips.txt",
    val_list=f"{work_dir}/val_clips.txt",
    batch_size=batch_size,
    num_workers=4,
    sequence_length=seq_length,
    shuffle=shuffle,
    persistent_workers=True,
    pin_memory=True,
    xrel_range=xrel_range,
    yrel_range=yrel_range,
    camera_groups=[CameraGroupConfig.front_stereo_camera_group(), CameraGroupConfig.short_focal_length_camera_group(), CameraGroupConfig.rear_camera_group()]
)


# ============================== 4. Model Config ==============================
def repvgg_backbone(name: str="a1", scales_to_drop: List[int]=[2, 4], use_pretrained: bool=True):
    return dict(
        type="ImageFeatureExtractor",
        backbone=f'repvgg_{name}',
        scales_to_drop=scales_to_drop,
        use_pretrained=use_pretrained
    )

def fpn_neck(in_channels: List[int]=[256, 512, 1024]):
    return dict(
        type="FPN",
        in_channels=in_channels,
        out_channels=query_dim,
        extra_blocks=0,
        relu_before_extra_convs=False
    )

lightning_module = dict(
    type="Sparse4DModule",
    detector=dict(
        type="Sparse4DDetector",
        camera_groups=dict(
            front_stereo_camera=[SourceCameraId.FRONT_LEFT_CAMERA, SourceCameraId.FRONT_RIGHT_CAMERA],
            short_focal_length_camera=[SourceCameraId.FRONT_CENTER_CAMERA, SourceCameraId.SIDE_LEFT_CAMERA, SourceCameraId.SIDE_RIGHT_CAMERA],
            rear_camera=[SourceCameraId.REAR_LEFT_CAMERA, SourceCameraId.REAR_RIGHT_CAMERA]
        ),
        anchor_encoder=dict(
            type="AnchorEncoder",
            pos_embed_dim=128,
            dim_embed_dim=32,
            orientation_embed_dim=32,
            vel_embed_dim=64,
            embed_dim=query_dim,
            anchor_generator_config=dict(
                type="Anchor3DGenerator",
                front_type="div_x",
                back_type="div_x",
                front_params=dict(alpha=0.28, beta=20.0, order=1.5), # beta * torch.pow(x, order) / (alpha - x), x = idx * min_spacing / max_distance
                back_params=dict(alpha=0.2, beta=20.0, order=1.5),
                front_min_spacing=2.0,
                front_max_distance=200.0,
                back_min_spacing=2.0,
                back_max_distance=100.0,
                left_y_max=3.75 * 2,
                right_y_max=3.75 * 2,
                y_interval=3.75,
                z_value=0.2,
                anchor_size=(5.0, 2.0, 1.5) # (length, width, height)
            ),
        ),
        feature_extractors=dict(
            # Ensure all the feature extractors have the same FPN levels
            front_stereo_camera=dict(
                type="FPNImageFeatureExtractor",
                backbone=repvgg_backbone(name="a2", scales_to_drop=[2, 4], use_pretrained=True),
                neck=fpn_neck()
            ),
            short_focal_length_camera=dict(
                type="FPNImageFeatureExtractor",
                backbone=repvgg_backbone(name="a1", scales_to_drop=[2], use_pretrained=True),
                neck=fpn_neck()
            ),
            rear_camera=dict(
                type="FPNImageFeatureExtractor",
                backbone=repvgg_backbone(name="a1", scales_to_drop=[2], use_pretrained=True),
                neck=fpn_neck()
            )
        ),
        decoder_op_orders= [
            [
                "mts_feature_aggregator",
                "ffn",
                "refine",
            ]] + [
            [
                "self_attention",
                "mts_feature_aggregator",
                "ffn",
                "refine",
            ]
            for _ in range(num_decoder - 1)
        ],
        mts_feature_aggregator=dict(
            type="MultiviewTemporalSpatialFeatureAggregator",
            query_dim=query_dim,
            num_learnable_points=8,
            learnable_points_range=3.0,
            sequence_length=seq_length,
            temporal_weight_decay=0.5,
            camera_nb=7,
            fpn_levels=3,
            residual_mode="cat"
        ),
        self_attention=dict(
            type="DecoupledMultiHeadAttention",
            query_dim=query_dim,
            num_heads=num_groups,
            dropout=dropout,
            post_norm=dict(type="LayerNorm", eps=1e-5, normalized_shape=query_dim),
        ),
        ffn=dict(
            type="AsymmetricFFN",
            pre_norm=dict(type="LayerNorm", eps=1e-5, normalized_shape=query_dim * 2),
            post_norm=dict(type="LayerNorm", eps=1e-5, normalized_shape=query_dim),
            in_channels=query_dim * 2,
            embed_dims=query_dim,
            feedforward_channels=query_dim * 4,
            num_fcs=2,
            act_cfg=dict(type="ReLU", inplace=True),
            ffn_drop=0.1,
            add_identity=True
        ),
        refine=dict(
            type="TrajectoryRefiner",
            query_dim=query_dim,
            hidden_dim=query_dim,
            with_quality_estimation=with_quality_estimation,
            motion_range=dict(
                x=xrel_range,
                y=yrel_range,
                z=[-3.0, 5.0],
                vx=[-40.0, 40.0],
                vy=[-5.0, 5.0],
                ax=[-5.0, 5.0],
                ay=[-2.0, 2.0]
            )
        ),
        temp_attention=dict(
            type="DecoupledMultiHeadAttention",
            query_dim=query_dim,
            num_heads=num_groups,
            dropout=dropout,
            post_norm=None
        ) if use_temp_attention else None
    ),
    scheduler=dict(
        type="CosineAnnealingLRScheduler",
        T_max=epochs,
        eta_min=0.0001
    ),
    optimizer=dict(
        type="AdamWOptimizer",
        lr=0.001,
        weight_decay=0.0001
    ),
    loss=dict(
        type="Sparse4DLossWithDAC",
        xrel_range=xrel_range,
        yrel_range=yrel_range,
        layer_loss_weights=[0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.9],
        has_object_loss=dict(
            type="FocalLoss",
            alpha=0.25,
            gamma=2.0,
            reduction="mean",
            loss_weight=1.0,
            pos_weight=[5.0]
        ),
        attribute_loss=dict(
            type="FocalLoss",
            alpha=[0.25 for _ in range(num_classes - 1)],
            gamma=2.0,
            reduction="mean",
            loss_weight=0.2,
            pos_weight=[1.0 for _ in range(num_classes - 1)]
        ),
        regression_loss=dict(
            type="SmoothL1Loss",
            beta=1.0,
            reduction="mean",
            loss_weight=1.0
        ),
    ),
    debug_config = dict(
        visualize_intermediate_results=True,
        visualize_intermediate_results_dir=f"{work_dir}/xinnovation_visualize_intermediate_results",
        visualize_camera_list=[SourceCameraId.FRONT_LEFT_CAMERA, SourceCameraId.REAR_LEFT_CAMERA],
        render_gt_trajs=True,
        render_init_trajs=False,
        render_pred_trajs=True,
        render_matched_trajs=True,
        pred_traj_threshold=0.5,
        render_trajs_interval=1, # every 10 epochs
        gt_color=[0.0, 255.0, 0.0],
        init_color=[0.0, 0.0, 255.0],
        pred_color=[255.0, 0.0, 0.0],
        point_radius=1,
        log_dir=f"{save_dir}/logs/{exp_name}",
        predict_dir=f"{save_dir}/predict/{exp_name}",
    ),
)

# ============================== 5. Evaluation Config ==============================



if __name__ == "__main__":
    from xinnovation.src.core.config import Config
    from easydict import EasyDict as edict
    import os
    
    # Get the absolute path to this file
    current_file = os.path.abspath(__file__)
    cfg = Config.from_file(current_file)
    config = Config.from_file(current_file).to_dict()
    cfg.save_to_file(current_file.replace(".py", ".json"))
    cfg.save_to_file(current_file.replace(".py", ".yaml"))
