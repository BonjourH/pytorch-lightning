import os
import torch
import lightning as L
import argparse
from typing import Dict, List
import json
import numpy as np

from base import (
    SourceCameraId, ObjectType, TrajParamIndex,
    tensor_to_trajectory, ObstacleTrajectory
)
from data import MultiFrameDataset, custom_collate_fn
from model import E2EPerceptionNet

class E2EPerceptionPredictor(L.LightningModule):
    """Lightning module for end-to-end perception inference."""
    def __init__(self, 
                 camera_ids: List[SourceCameraId],
                 feature_dim: int = 256,
                 num_queries: int = 100,
                 num_decoder_layers: int = 6,
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
            num_decoder_layers=num_decoder_layers
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
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--feature-dim', type=int, default=256, help='Feature dimension')
    parser.add_argument('--num-queries', type=int, default=100, help='Number of object queries')
    parser.add_argument('--num-decoder-layers', type=int, default=6, help='Number of decoder layers')
    
    # Data parameters
    parser.add_argument('--test-list', type=str, required=True, help='Path to txt file containing test clip paths')
    parser.add_argument('--sequence-length', type=int, default=10, help='Number of frames in each sequence')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results')
    
    # Inference parameters
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--confidence-threshold', type=float, default=0.5, help='Confidence threshold for predictions')
    
    # Trainer parameters
    parser.add_argument('--accelerator', type=str, default='auto', help='Accelerator to use (auto, gpu, cpu)')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices to use')
    parser.add_argument('--precision', type=str, default='32-true', help='Precision for training')
    
    return parser.parse_args()

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

def main():
    args = parse_args()
    
    # Define camera IDs
    camera_ids = [
        SourceCameraId.FRONT_CENTER_CAMERA,
        SourceCameraId.FRONT_LEFT_CAMERA,
        SourceCameraId.FRONT_RIGHT_CAMERA,
        SourceCameraId.SIDE_LEFT_CAMERA,
        SourceCameraId.SIDE_RIGHT_CAMERA,
        SourceCameraId.REAR_LEFT_CAMERA,
        SourceCameraId.REAR_RIGHT_CAMERA
    ]
    
    # Create predictor
    predictor = E2EPerceptionPredictor(
        camera_ids=camera_ids,
        feature_dim=args.feature_dim,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_list=args.test_list,
        confidence_threshold=args.confidence_threshold,
        output_dir=args.output_dir
    )
    
    # Create trainer
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=False  # Disable logging for inference
    )
    
    # Load checkpoint and run inference
    trainer.predict(
        predictor,
        ckpt_path=args.checkpoint
    )

if __name__ == '__main__':
    main() 