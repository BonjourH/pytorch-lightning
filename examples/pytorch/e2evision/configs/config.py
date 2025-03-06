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
class DataConfig(ConfigBase):
    """Data configuration."""
    train_list: str = "train_clips.txt"
    val_list: str = "val_clips.txt"
    test_list: str = "test_list.txt"
    sequence_length: int = 10
    batch_size: int = 1
    num_workers: int = 4
    image_size: tuple = (800, 416)
    normalize_mean: tuple = (0.485, 0.456, 0.406)
    normalize_std: tuple = (0.229, 0.224, 0.225)
    camera_ids: List[SourceCameraId] = field(default_factory=lambda: [
        SourceCameraId.FRONT_CENTER_CAMERA,
        SourceCameraId.FRONT_LEFT_CAMERA,
        SourceCameraId.FRONT_RIGHT_CAMERA,
        SourceCameraId.SIDE_LEFT_CAMERA,
        SourceCameraId.SIDE_RIGHT_CAMERA,
        SourceCameraId.REAR_LEFT_CAMERA,
        SourceCameraId.REAR_RIGHT_CAMERA
    ])


@dataclass
class ModelConfig(ConfigBase):
    """Model configuration"""
    feature_dim: int = 256
    num_queries: int = 16
    num_decoder_layers: int = 6
    backbone: str = 'resnet18'


@dataclass
class TrainingConfig(ConfigBase):
    """Training configuration"""
    train_list: str = 'train_clips.txt'
    val_list: str = 'val_clips.txt'
    batch_size: int = 1
    num_workers: int = 1
    max_epochs: int = 100
    accelerator: str = 'gpu'
    devices: int = 1
    precision: str = '16-mixed'
    accumulate_grad_batches: int = 4
    seed: int = 42
    pretrained_weights: bool = True
    # 优化器配置
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip_val: float = 0.5


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


@dataclass
class LoggingConfig(ConfigBase):
    """Logging and checkpoint configuration"""
    log_dir: str = 'logs'
    results_dir: str = 'results'
    checkpoint_dir: str = 'checkpoints'
    checkpoint_file: str = 'model_last.ckpt'


@dataclass
class Config(ConfigBase):
    """Collection of all configurations"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Config':
        """Create configuration object from dictionary"""
        return cls(
            data=DataConfig(**data.get('data', {})),
            model=ModelConfig(**data.get('model', {})),
            training=TrainingConfig(**data.get('training', {})),
            inference=InferenceConfig(**data.get('inference', {})),
            logging=LoggingConfig(**data.get('logging', {}))
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
