import os
import torch
import argparse
from tqdm import tqdm
import logging
from typing import Dict, List
import json
import numpy as np

from base import (
    SourceCameraId, ObjectType, TrajParamIndex,
    tensor_to_trajectory, ObstacleTrajectory
)
from data import MultiFrameDataset, custom_collate_fn
from model import E2EPerceptionNet

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with trained model')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--feature-dim', type=int, default=256,
                       help='Feature dimension')
    parser.add_argument('--num-queries', type=int, default=100,
                       help='Number of object queries')
    parser.add_argument('--num-decoder-layers', type=int, default=6,
                       help='Number of decoder layers')
    
    # Data parameters
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory containing test clips')
    parser.add_argument('--sequence-length', type=int, default=10,
                       help='Number of frames in each sequence')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results')
    
    # Inference parameters
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                       help='Confidence threshold for predictions')
    
    return parser.parse_args()

def setup_logging(output_dir: str):
    """Setup logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'inference.log')),
            logging.StreamHandler()
        ]
    )

def load_model(checkpoint_path: str, args, device: str) -> torch.nn.Module:
    """Load model from checkpoint."""
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
    
    # Create model
    model = E2EPerceptionNet(
        camera_ids=camera_ids,
        feature_dim=args.feature_dim,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    logging.info(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
    return model

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

def save_results(
    trajectories: List[ObstacleTrajectory],
    timestamp: float,
    output_path: str
):
    """Save prediction results to JSON file."""
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

def main():
    args = parse_args()
    setup_logging(args.output_dir)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model(args.checkpoint, args, device)
    model.eval()
    
    # Create test dataset
    test_clips = []  # TODO: Get list of test clips from data_root
    for item in os.listdir(args.data_root):
        item_path = os.path.join(args.data_root, item)
        if os.path.isdir(item_path):
            test_clips.append(item_path)
    
    test_dataset = MultiFrameDataset(
        clip_dirs=test_clips,
        camera_ids=model.camera_ids,
        sequence_length=args.sequence_length
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    # Run inference
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Inference')):
            # Move data to device
            for camera_id in batch['images']:
                batch['images'][camera_id] = batch['images'][camera_id].to(device)
            for camera_id in batch['calibrations']:
                batch['calibrations'][camera_id] = batch['calibrations'][camera_id].to(device)
            batch['ego_states'] = batch['ego_states'].to(device)
            
            # Forward pass
            outputs = model(batch)
            
            # Process predictions
            trajectories = process_predictions(
                outputs,
                args.confidence_threshold
            )
            
            # Save results
            timestamp = batch_idx * args.sequence_length * 0.1  # Assuming 10Hz
            output_path = os.path.join(
                args.output_dir,
                f'predictions_{batch_idx:06d}.json'
            )
            save_results(trajectories, timestamp, output_path)
            
            # Log some statistics
            if batch_idx % 10 == 0:
                logging.info(f'Processed batch {batch_idx}, '
                           f'found {len(trajectories)} objects')

if __name__ == '__main__':
    main() 