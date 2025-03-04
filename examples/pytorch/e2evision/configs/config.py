from dataclasses import dataclass
from typing import List, Optional
from base import SourceCameraId
import torch

@dataclass
class DataConfig:
    """Data configuration."""
    train_list: str
    val_list: str
    sequence_length: int = 10
    batch_size: int = 2
    num_workers: int = 4
    image_size: tuple = (224, 224)
    normalize_mean: tuple = (0.485, 0.456, 0.406)
    normalize_std: tuple = (0.229, 0.224, 0.225)
    camera_ids: List[SourceCameraId] = None

@dataclass
class ModelConfig:
    """Model configuration."""
    feature_dim: int = 256
    num_queries: int = 100
    num_decoder_layers: int = 6
    hidden_dim: int = 512
    dropout: float = 0.1
    num_attention_heads: int = 8

@dataclass
class TrainingConfig:
    """Training configuration."""
    max_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    lr_scheduler: str = 'cosine'  # ['cosine', 'step', 'plateau']
    gradient_clip_val: float = 0.1
    precision: str = '32-true'
    accumulate_grad_batches: int = 1
    check_val_every_n_epoch: int = 1
    log_every_n_steps: int = 10

@dataclass
class LoggingConfig:
    """Logging configuration."""
    save_dir: str = 'checkpoints'
    experiment_name: str = 'e2e_perception'
    save_top_k: int = 3
    monitor: str = 'val_loss'
    monitor_mode: str = 'min'
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

@dataclass
class Config:
    """Main configuration."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig
    seed: int = 42
    accelerator: str = 'auto'
    devices: int = 1
    strategy: str = 'auto'

    def __post_init__(self):
        """Set default camera IDs if not specified."""
        if self.data.camera_ids is None:
            self.data.camera_ids = [
                SourceCameraId.FRONT_CENTER_CAMERA,
                SourceCameraId.FRONT_LEFT_CAMERA,
                SourceCameraId.FRONT_RIGHT_CAMERA,
                SourceCameraId.SIDE_LEFT_CAMERA,
                SourceCameraId.SIDE_RIGHT_CAMERA,
                SourceCameraId.REAR_LEFT_CAMERA,
                SourceCameraId.REAR_RIGHT_CAMERA
            ]

    def save(self, path: str):
        """Save configuration to file."""
        import json
        from dataclasses import asdict
        
        def serialize(obj):
            if isinstance(obj, SourceCameraId):
                return obj.name
            return str(obj)
        
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=serialize)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from file."""
        import json
        
        def deserialize_camera_id(name: str) -> SourceCameraId:
            return SourceCameraId[name]
        
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Convert camera IDs back to enum
        if 'data' in data and 'camera_ids' in data['data']:
            data['data']['camera_ids'] = [
                deserialize_camera_id(name) for name in data['data']['camera_ids']
            ]
            
        # Create configs
        data_config = DataConfig(**data['data'])
        model_config = ModelConfig(**data['model'])
        training_config = TrainingConfig(**data['training'])
        logging_config = LoggingConfig(**data['logging'])
        
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            logging=logging_config,
            seed=data.get('seed', 42),
            accelerator=data.get('accelerator', 'auto'),
            devices=data.get('devices', 1),
            strategy=data.get('strategy', 'auto')
        ) 