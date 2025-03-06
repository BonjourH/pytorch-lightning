import os
import torch
import lightning as L
import argparse
from typing import Dict, List
import json
import numpy as np
import sys

from base import (
    SourceCameraId, ObjectType, TrajParamIndex,
    tensor_to_trajectory, ObstacleTrajectory
)
from data import MultiFrameDataset, custom_collate_fn
from model import E2EPerceptionNet

# 导入配置文件
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.pytorch.e2evision.configs.default_config import get_config, update_config

class E2EPerceptionPredictor(L.LightningModule):
    """Lightning module for end-to-end perception inference."""
    def __init__(self, 
                 camera_ids: List[SourceCameraId],
                 feature_dim: int = 256,
                 num_queries: int = 100,
                 num_decoder_layers: int = 6,
                 backbone: str = 'resnet18',
                 sequence_length: int = 10,
                 batch_size: int = 1,
                 num_workers: int = 4,
                 test_list: str = None,
                 confidence_threshold: float = 0.5,
                 output_dir: str = 'results'):
        super().__init__()
        self.save_hyperparameters(ignore=['camera_ids'])
        self.camera_ids = camera_ids
        
        # Create model
        self.model = E2EPerceptionNet(
            camera_ids=camera_ids,
            feature_dim=feature_dim,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            backbone=backbone
        )
        
    def forward(self, batch: Dict) -> List[Dict]:
        return self.model(batch)
    
    def predict_step(self, batch: Dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Forward pass
        outputs = self(batch)
        
        # Process predictions
        trajectories = self.process_predictions(
            outputs,
            self.hparams.confidence_threshold,
            timestamp=batch_idx * self.hparams.sequence_length * 0.1  # Assuming 10Hz
        )
        
        # Save results
        output_path = os.path.join(
            self.hparams.output_dir,
            f'predictions_{batch_idx:06d}.json'
        )
        self.save_results(trajectories, batch_idx * self.hparams.sequence_length * 0.1, output_path)
        
        # Log some statistics
        if batch_idx % 10 == 0:
            self.print(f'Processed batch {batch_idx}, found {len(trajectories)} objects')
    
    def setup(self, stage: str):
        """Setup datasets and dataloaders."""
        if stage == "predict":
            # Read test clip list
            test_clips = read_clip_list(self.hparams.test_list)
            self.print(f"Found {len(test_clips)} test clips")
            
            # Create dataset
            self.test_dataset = MultiFrameDataset(
                clip_dirs=test_clips,
                camera_ids=self.camera_ids,
                sequence_length=self.hparams.sequence_length
            )
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True
        )
    
    @staticmethod
    def process_predictions(
        outputs: List[Dict],
        confidence_threshold: float,
        timestamp: float = 0.0
    ) -> List[ObstacleTrajectory]:
        """Process model outputs into trajectory objects."""
        # Get final predictions (last element contains final predictions)
        final_pred = outputs[-1]
        pred_trajs = final_pred['traj_params']  # [B, N, TrajParamIndex.END_OF_INDEX]
        type_logits = final_pred['type_logits']  # [B, N, len(ObjectType)]
        
        # Get existence probabilities
        existence_probs = torch.sigmoid(pred_trajs[..., TrajParamIndex.HAS_OBJECT])
        
        # Filter predictions by confidence threshold
        valid_mask = existence_probs[0] > confidence_threshold  # Use first batch
        valid_trajs = pred_trajs[0][valid_mask]  # [valid_N, TrajParamIndex.END_OF_INDEX]
        valid_type_logits = type_logits[0][valid_mask]  # [valid_N, len(ObjectType)]
        
        # Convert predictions to trajectory objects
        trajectories = []
        for i, (traj_params, type_logits) in enumerate(zip(valid_trajs, valid_type_logits)):
            # Convert tensor parameters to trajectory object
            trajectory = tensor_to_trajectory(
                traj_params=traj_params,
                traj_id=i,
                t0=timestamp
            )
            trajectories.append(trajectory)
        
        return trajectories
    
    @staticmethod
    def save_results(
        trajectories: List[ObstacleTrajectory],
        timestamp: float,
        output_path: str
    ):
        """Save prediction results to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results = {
            'timestamp': timestamp,
            'obstacles': []
        }
        
        for traj in trajectories:
            obstacle = {
                'id': traj.id,
                'x': float(traj.position[0]),
                'y': float(traj.position[1]),
                'z': float(traj.position[2]),
                'vx': float(traj.velocity[0]),
                'vy': float(traj.velocity[1]),
                'ax': float(traj.acceleration[0]),
                'ay': float(traj.acceleration[1]),
                'yaw': float(traj.yaw),
                'length': float(traj.length),
                'width': float(traj.width),
                'height': float(traj.height),
                'type': traj.object_type.name,
                'static': traj.static,
                'valid': traj.valid
            }
            results['obstacles'].append(obstacle)
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with trained model')
    
    # 配置文件参数
    parser.add_argument('--config-module', type=str, default=None, 
                        help='Python module with custom config (e.g. configs.my_config)')
    
    # 重要的控制参数
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--test-list', type=str, help='Path to txt file containing test clip paths')
    
    # 通用方式覆盖配置文件中的任意配置项
    parser.add_argument('--config-override', nargs='+', action='append', 
                        help='Override config values. Format: section.key=value')
    
    return parser.parse_args()

def handle_config_overrides(config, args):
    """处理命令行中的配置覆盖"""
    if not args.config_override:
        return config
    
    for override in args.config_override:
        for item in override:
            path, value = item.split('=', 1)
            sections = path.split('.')
            
            # 尝试转换值为适当的类型
            try:
                value = eval(value)  # 尝试将字符串解析为Python对象
            except:
                pass  # 如果失败，保持为字符串
            
            # 处理嵌套配置
            current = config
            for section in sections[:-1]:
                if section not in current:
                    current[section] = {}
                current = current[section]
            
            # 设置最终值
            current[sections[-1]] = value
    
    return config

def load_python_config(config_module_path):
    """从Python模块加载配置"""
    import importlib
    try:
        config_module = importlib.import_module(config_module_path)
        return config_module.get_config()
    except (ImportError, AttributeError) as e:
        print(f"Error loading config from {config_module_path}: {e}")
        return None

def read_clip_list(list_file: str) -> List[str]:
    """Read clip paths from a txt file."""
    if not os.path.exists(list_file):
        raise FileNotFoundError(f"Clip list file not found: {list_file}")
        
    with open(list_file, 'r') as f:
        clips = [line.strip() for line in f.readlines() if line.strip()]
        
    # Verify all paths exist
    for clip_path in clips:
        if not os.path.exists(clip_path):
            raise FileNotFoundError(f"Clip directory not found: {clip_path}")
            
    return clips

# Main function
if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    if args.config_module:
        # 如果指定了自定义配置模块，加载它
        custom_config = load_python_config(args.config_module)
        if custom_config:
            config = update_config(get_config(), custom_config)
        else:
            config = get_config()
    else:
        # 否则使用默认配置
        config = get_config()
    
    # 应用命令行覆盖
    config = handle_config_overrides(config, args)
    
    # 设置特定参数（如果在命令行中指定）
    test_list = args.test_list if args.test_list else config['inference']['test_list']
    output_dir = args.output_dir if args.output_dir else config['logging']['results_dir']
    checkpoint_path = args.checkpoint if args.checkpoint else os.path.join(config['logging']['checkpoint_dir'], config['logging']['checkpoint_file'])
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建相机ID列表
    camera_ids = [SourceCameraId(camera_id) for camera_id in config['cameras']['camera_ids']]
    
    # 读取测试列表
    test_clips = read_clip_list(test_list)
    
    # 创建预测器
    predictor = E2EPerceptionPredictor(
        camera_ids=camera_ids,
        feature_dim=config['model']['feature_dim'],
        num_queries=config['model']['num_queries'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        backbone=config['model']['backbone'],
        sequence_length=config['inference']['sequence_length'],
        batch_size=config['inference']['batch_size'],
        num_workers=config['inference']['num_workers'],
        test_list=test_list,
        confidence_threshold=config['inference']['confidence_threshold'],
        output_dir=output_dir
    )
    
    # 创建训练器
    trainer = L.Trainer(
        accelerator=config['inference']['accelerator'],
        devices=config['inference']['devices'],
        precision=config['inference']['precision']
    )
    
    # 运行预测
    print(f"Running inference on {len(test_clips)} clips with checkpoint: {checkpoint_path}")
    trainer.predict(predictor, ckpt_path=checkpoint_path) 