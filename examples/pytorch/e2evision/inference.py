import os
import torch
import lightning as L
import argparse
from typing import Dict, List
import json
import numpy as np
import sys
import time
from torch.utils.data import DataLoader

from base import (
    SourceCameraId, ObjectType, TrajParamIndex,
    tensor_to_trajectory, ObstacleTrajectory
)
from e2e_dataset.dataset import MultiFrameDataset, custom_collate_fn
from models.module import E2EPerceptionModule
from models.network import E2EPerceptionNet
from configs.config import get_config, Config

class E2EPerceptionPredictor(L.LightningModule):
    """Lightning module for end-to-end perception inference."""
    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])
        self.config = config
        
        # Create model
        self.model = E2EPerceptionModule(config=config)
        
    def forward(self, batch: Dict) -> List[Dict]:
        return self.model(batch)
    
    def predict_step(self, batch: Dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Forward pass
        outputs = self(batch)
        
        # Process predictions
        trajectories = self.process_predictions(
            outputs,
            self.config.inference.confidence_threshold,
            timestamp=batch_idx * self.config.data.sequence_length * 0.1  # Assuming 10Hz
        )
        
        # Save results
        output_path = os.path.join(
            self.config.inference.output_dir,
            f'predictions_{batch_idx:06d}.json'
        )
        self.save_results(trajectories, batch_idx * self.config.data.sequence_length * 0.1, output_path)
        
        # Log some statistics
        if batch_idx % 10 == 0:
            self.print(f'Processed batch {batch_idx}, found {len(trajectories)} objects')
    
    def setup_predict_dataloader(self, test_list: str):
        """Setup predict dataloader."""
        # Read test clip list
        test_clips = self.read_clip_list(test_list)
        self.print(f"Found {len(test_clips)} test clips")
        
        # Create dataset
        test_dataset = MultiFrameDataset(
            clip_dirs=test_clips,
            config=self.config.data
        )
        
        return DataLoader(
            test_dataset,
            batch_size=self.config.inference.batch_size,
            shuffle=False,
            num_workers=self.config.inference.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True
        )
    
    def read_clip_list(self, list_file: str) -> List[str]:
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
    parser.add_argument('--config_file', type=str, default=None, help='config file path, could be a python file, yaml file or json file')
    
    # 重要的控制参数
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--test-list', type=str, help='Path to txt file containing test clip paths')
    
    # 通用方式覆盖配置文件中的任意配置项
    parser.add_argument('--config-override', nargs='+', action='append', 
                        help='Override config values. Format: section.key=value')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 获取配置
    config = get_config(args)
    
    # 设置随机种子
    L.seed_everything(config.inference.seed, workers=True)
    
    # 设置输出目录
    if args.output_dir:
        config.inference.output_dir = args.output_dir
    
    # 创建输出目录
    os.makedirs(config.inference.output_dir, exist_ok=True)
    
    # 创建模型
    model = E2EPerceptionPredictor(config=config)
    
    # 加载检查点
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(config.logging.checkpoint_dir, 'last.ckpt')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # 创建推理器
    trainer = L.Trainer(
        accelerator=config.inference.accelerator,
        devices=config.inference.devices,
        precision=config.inference.precision,
        logger=False,
        enable_progress_bar=True,
    )
    
    # 设置测试数据加载器
    if args.test_list:
        test_list = args.test_list
    else:
        test_list = config.inference.test_list
    
    predict_dataloader = model.setup_predict_dataloader(test_list)
    
    # 运行推理
    print(f"Running inference with checkpoint: {checkpoint_path}")
    print(f"Output directory: {config.inference.output_dir}")
    
    trainer.predict(model, predict_dataloader, ckpt_path=checkpoint_path)
    
    print(f"Inference completed. Results saved to {config.inference.output_dir}")

if __name__ == '__main__':
    main() 