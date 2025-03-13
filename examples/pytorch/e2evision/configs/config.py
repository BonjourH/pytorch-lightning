from dataclasses import dataclass, field, asdict, fields
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import json
import importlib
import argparse
from pathlib import Path
import yaml
from easydict import EasyDict
from base import SourceCameraId  


class ConfigBase:
    def to_dict(self) -> Dict[str, Any]:
        """Recursive conversion to dictionary"""
        return asdict(self)

    def update(self, updates: Dict[str, Any]):
        """Recursive update configuration"""
        import pdb; pdb.set_trace()
        def _update(target, key_path, value):
            keys = key_path.split('.')
            for key in keys[:-1]:
                target = getattr(target, key)
            setattr(target, keys[-1], value)

        for key_path, value in flatten_dict(updates):
            _update(self, key_path, value)

@dataclass
class CameraGroupConfig(ConfigBase):
    """Camera group configuration"""
    name: str = ''
    camera_group: List[SourceCameraId] = field(default_factory=list)
    image_size: tuple = (800, 416)
    normalize_mean: tuple = (0.485, 0.456, 0.406)
    normalize_std: tuple = (0.229, 0.224, 0.225)
    backbone: str = 'resnet18'
    out_channels: int = 256
    fpn_channels: int = 256
    downsample_scale: int = 32
    use_pretrained: bool = True
    
    @classmethod
    def long_focal_length_camera_group(cls):
        return cls(name='long_focal_length_camera', 
                   camera_group=[SourceCameraId.FRONT_LEFT_CAMERA, SourceCameraId.FRONT_RIGHT_CAMERA,
                                 SourceCameraId.REAR_LEFT_CAMERA, SourceCameraId.REAR_RIGHT_CAMERA],
                   image_size=(576, 288),
                   backbone='resnet18',
                   downsample_scale=8)
    
    @classmethod
    def front_stereo_camera_group(cls):
        return cls(name='front_stereo_camera', 
                   camera_group=[SourceCameraId.FRONT_LEFT_CAMERA, SourceCameraId.FRONT_RIGHT_CAMERA],
                   image_size=(576, 288),
                   backbone='repvgg_a1',
                   downsample_scale=4)
    
    @classmethod
    def short_focal_length_camera_group(cls):
        return cls(name='short_focal_length_camera', 
                   camera_group=[SourceCameraId.SIDE_LEFT_CAMERA, SourceCameraId.SIDE_RIGHT_CAMERA, SourceCameraId.FRONT_CENTER_CAMERA],
                   image_size=(400, 256),
                   backbone='repvgg_a1',
                   downsample_scale=16)
    
    @classmethod    
    def rear_camera_group(cls):
        return cls(name='rear_camera', 
                   camera_group=[SourceCameraId.REAR_LEFT_CAMERA, SourceCameraId.REAR_RIGHT_CAMERA],
                   image_size=(576, 288),
                   backbone='repvgg_a1',
                   downsample_scale=8)

@dataclass
class DataConfig(ConfigBase):
    """Data configuration."""
    train_list: str = "train_clips.txt"
    val_list: str = "val_clips.txt"
    test_list: str = "test_list.txt"
    sequence_length: int = 8
    shuffle: bool = True
    persistent_workers: bool = False
    batch_size: int = 5
    num_workers: int = 20
    camera_ids: List[SourceCameraId] = field(default_factory=lambda: [
        SourceCameraId.FRONT_CENTER_CAMERA,
        SourceCameraId.FRONT_LEFT_CAMERA,
        SourceCameraId.FRONT_RIGHT_CAMERA,
        SourceCameraId.SIDE_LEFT_CAMERA,
        SourceCameraId.SIDE_RIGHT_CAMERA,
        SourceCameraId.REAR_LEFT_CAMERA,
        SourceCameraId.REAR_RIGHT_CAMERA
    ])
    camera_groups: List[CameraGroupConfig] = field(default_factory=lambda: [
        CameraGroupConfig.front_stereo_camera_group(),
        CameraGroupConfig.short_focal_length_camera_group(),
        CameraGroupConfig.rear_camera_group()
    ])

@dataclass
class DecoderConfig(ConfigBase):
    """Decoder configuration"""
    num_layers: int = 3
    num_queries: int = 128
    feature_dim: int = 256
    hidden_dim: int = 512
    num_points: int = 25

@dataclass
class ModelConfig(ConfigBase):
    """Model configuration"""
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

@dataclass
class TrainingConfig(ConfigBase):
    """Training configuration"""
    train_list: str = 'train_clips.txt'
    val_list: str = 'val_clips.txt'
    batch_size: int = 1
    num_workers: int = 1
    max_epochs: int = 1000
    accelerator: str = 'gpu'
    devices: int = 1
    precision: str = '16-mixed'
    accumulate_grad_batches: int = 4
    seed: int = 42
    pretrained_weights: bool = False
    limit_val_batches: float = 1  # 1: one batch, 0.1: validate 10% of batches, 1.0: validate all batches
    limit_train_batches: float = 1  # 1: one batch, 0.1: validate 10% of batches, 1.0: validate all batches
    # 优化器配置
    learning_rate: float = 1e-4
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 1000.0
    weight_decay: float = 1e-4
    gradient_clip_val: float = 1.0
    use_checkpoint: bool = True
    log_every_n_steps: int = 30
    check_val_every_n_epoch: int = 5
    num_sanity_val_steps: int = 0 # 0: no sanity check, 1: check 1 batch, -1: check all batches

@dataclass
class InferenceConfig(ConfigBase):
    """Inference configuration"""
    test_list: str = 'test_list.txt'
    confidence_threshold: float = 0.5
    batch_size: int = 1
    num_workers: int = 4
    accelerator: str = 'gpu'
    devices: int = 1
    precision: str = '16-mixed'
    output_dir: str = 'results'
    seed: int = 42
    checkpoint: str = 'checkpoints/last.ckpt'
    limit_batch_size: int = 1


@dataclass
class LoggingConfig(ConfigBase):
    """Logging and checkpoint configuration"""
    log_dir: str = 'logs'
    checkpoint_dir: str = 'checkpoints'
    save_top_k: int = 3
    last_checkpoint_dir: Optional[str] = None 
    # Logger options
    use_tensorboard: bool = True
    use_csv: bool = True
    use_wandb: bool = True
    wandb_project: str = 'e2e_perception'
    run_id: Optional[int] = 0
    clean_wandb_history: bool = True  # 是否清理 W&B 上的历史数据（只保留当前训练）
    # Progress bar options
    progress_bar_metrics: List[str] = field(default_factory=lambda: [
        'v_num', 
        'train/loss_step', 
        'train/loss_epoch',
        'train/layer_3_loss_cls_epoch',
        'train/layer_3_loss_pos_epoch',
        'val/loss',
        'epoch',
        'step'
    ])
    # Wandb logger options
    use_optional_metrics: bool = True
    wandb_log_metrics: List[str] = field(default_factory=lambda: [
        'train/loss_epoch', 
        'val/loss',
        'train/layer_3_loss_pos_epoch',
        'train/layer_3_loss_dim_epoch',
        'train/layer_3_loss_vel_epoch',
        'train/layer_3_loss_yaw_epoch',
        'train/layer_3_loss_acc_epoch',
        'train/layer_3_loss_cls_epoch',
        'train/layer_3_fp_loss_exist_epoch',
        'train/layer_2_loss_cls_epoch',
        'train/layer_2_loss_pos_epoch',
        'train/layer_2_loss_dim_epoch',
        'train/layer_2_loss_vel_epoch',
        'train/layer_2_loss_yaw_epoch',
        'train/layer_2_loss_acc_epoch',
        'train/layer_2_loss_cls_epoch',
        'train/layer_2_fp_loss_exist_epoch',
        'train/layer_1_loss_cls_epoch',
        'train/layer_1_loss_pos_epoch',
        'train/layer_1_loss_dim_epoch',
        'train/layer_1_loss_vel_epoch',
        'train/layer_1_loss_yaw_epoch',
        'train/layer_1_loss_acc_epoch',
        'train/layer_1_loss_cls_epoch',
        'train/layer_1_fp_loss_exist_epoch'
    ])
    # for intermediate results
    visualize_intermediate_results: bool = True
    visualize_intermediate_results_dir: str = 'visualize_intermediate_results'
    point_radius: int = 1 # muse be an odd number
    
@dataclass
class LossConfig(ConfigBase):
    """Loss configuration"""
    weight_dict: Dict[str, float] = field(default_factory=lambda: {
        'loss_pos': 1.0,
        'loss_dim': 1.0,
        'loss_vel': 0.5,
        'loss_yaw': 0.5,
        'loss_acc': 0.1,
        'loss_cls': 1.0,
        'fp_loss_exist': 2.0
    })
    layer_loss_weights: List[float] = field(default_factory=lambda: [
        0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0
    ])
    frames: int = 10
    dt: float = 0.1
    iou_method: str = "iou2"
    iou_threshold: float = 0.5


@dataclass
class Config(ConfigBase):
    """Collection of all configurations"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    predict: InferenceConfig = field(default_factory=InferenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Config':
        """Create configuration object from dictionary"""
        return cls(
            data=DataConfig(**data.get('data', {})),
            model=ModelConfig(**data.get('model', {})),
            training=TrainingConfig(**data.get('training', {})),
            predict=InferenceConfig(**data.get('predict', {})),
            logging=LoggingConfig(**data.get('logging', {})),
            loss=LossConfig(**data.get('loss', {}))
        )

    def save(self, path: Union[str, Path], format: str = 'json'):
        """Save configuration file"""
        path = Path(path)
        if format == 'json':
            with path.open('w') as f:
                json.dump(self.to_dict(), f, indent=2, default=self._serialize)
        elif format == 'yaml':
            with path.open('w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def _serialize(obj):
        """Custom serialization handling"""
        if isinstance(obj, Enum):
            return obj.name
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not serializable")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Config':
        """Load configuration from file"""
        try:
            path = Path(path)
            if path.suffix == '.json':
                return cls.load_from_json(path)
            elif path.suffix in ('.yaml', '.yml'):
                return cls.load_from_yaml(path)
            elif path.suffix == '.py':
                return cls.load_from_module(path)
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")
        except Exception as e:
            return Config()
    
    @classmethod
    def load_from_json(cls, path: Union[str, Path]) -> 'Config':
        """Load from JSON file"""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def load_from_yaml(cls, path: Union[str, Path]) -> 'Config':
        """Load from YAML file"""
        with open(path, 'r') as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def load_from_module(cls, path: Union[str, Path]) -> 'Config':
        """Load from Python module"""
        try:
            module = importlib.import_module(path)
            config_dict = {k: v for k, v in vars(module).items() 
                          if not k.startswith('_') and not callable(v)}
            return cls.from_dict(config_dict)
        except Exception as e:
            raise ImportError(f"Failed to load config from {path}: {str(e)}")

def deep_update(source, updates):
    """
    Recursively update dictionary, keep keys in source that are not in updates
    """
    for key, value in updates.items():
        if key in source and isinstance(source[key], dict) and isinstance(value, dict):
            # if both are dict, recursive update
            source[key] = deep_update(source[key], value)
        else:
            # otherwise, directly update or add
            source[key] = value
    return source

def update_dict_by_override_args(config: Dict[str, Any], args: argparse.Namespace) -> EasyDict:
    updates = {}
    if hasattr(args, 'config_override') and args.config_override:
        for override in args.config_override:
            for item in override:
                path, value = item.split('=', 1)
                
                try:
                    value = eval(value) 
                except:
                    pass
                
                current = updates
                parts = path.split('.')
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                current[parts[-1]] = value
    
    if hasattr(args, 'experiment_name') and args.experiment_name:
        pass
    
    return EasyDict(deep_update(config, updates))


def get_config(args: argparse.Namespace) -> EasyDict:
    config_dict = Config.load(args.config_file).to_dict()
    return update_dict_by_override_args(config_dict, args)
