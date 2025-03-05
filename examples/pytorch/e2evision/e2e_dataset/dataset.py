import os
import json
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image
from enum import IntEnum
from base import (
    SourceCameraId, CameraType, ObjectType,
    TrajParamIndex, CameraParamIndex,
    EgoStateIndex
)
import numpy as np


MAX_TRAJ_NB = 128

class TrainingSample:
    """Container for multi-frame training data."""
    def __init__(self, calibrations: Dict[SourceCameraId, torch.Tensor]):
        self.calibrations = calibrations
        self.trajs: Optional[torch.Tensor] = None
        # the below lists are of length T
        self.image_paths: List[Dict[SourceCameraId, str]] = []
        self.ego_states: List[torch.Tensor] = []
    
    def add_frame(self, 
                 image_paths: Dict[SourceCameraId, str],
                 ego_state: torch.Tensor):
        self.image_paths.append(image_paths)
        self.ego_states.append(ego_state)
    

class MultiFrameDataset(Dataset):
    """Dataset for multi-frame multi-camera perception."""
    def __init__(self, 
                 clip_dirs: List[str],
                 camera_ids: List[SourceCameraId],
                 sequence_length: int = 10,
                 transform=None):
        self.clip_dirs = clip_dirs
        self.camera_ids = camera_ids
        self.sequence_length = sequence_length
        self.transform = transform or transforms.Compose([
            transforms.Resize((416, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.samples = self._build_samples()
        print(f"Total {len(self.samples)} samples")

    def _build_samples(self) -> List[TrainingSample]:
        """Build list of training samples from clips."""
        samples = []
        
        for clip_dir in self.clip_dirs:
            # 1. Load calibrations
            calibrations: Dict[SourceCameraId, torch.Tensor] = {}
            for camera_id in self.camera_ids:
                calib_file = os.path.join(clip_dir, 'calib_json', f'{camera_id.name.lower()}.json')
                with open(calib_file, 'r') as f:
                    calib_data = json.load(f)
                    
                    # Create camera parameter vector
                    camera_params = torch.zeros(CameraParamIndex.END_OF_INDEX)
                    
                    # Set camera ID
                    camera_params[CameraParamIndex.CAMERA_ID] = camera_id
                    
                    # Set camera type
                    camera_type = CameraType[calib_data.get('type', 'PINHOLE')]
                    camera_params[CameraParamIndex.CAMERA_TYPE] = camera_type
                    
                    # Set image dimensions
                    camera_params[CameraParamIndex.IMAGE_WIDTH] = calib_data['intrinsic']['width']
                    camera_params[CameraParamIndex.IMAGE_HEIGHT] = calib_data['intrinsic']['height']
                    
                    # Set intrinsic parameters
                    camera_params[CameraParamIndex.FX] = calib_data['intrinsic']['fx']
                    camera_params[CameraParamIndex.FY] = calib_data['intrinsic']['fy']
                    camera_params[CameraParamIndex.CX] = calib_data['intrinsic']['cx']
                    camera_params[CameraParamIndex.CY] = calib_data['intrinsic']['cy']
                    
                    # Set distortion parameters based on camera type
                    if camera_type == CameraType.FISHEYE:
                        camera_params[CameraParamIndex.K1] = calib_data['intrinsic'].get('k1', 0.0)
                        camera_params[CameraParamIndex.K2] = calib_data['intrinsic'].get('k2', 0.0)
                        camera_params[CameraParamIndex.K3] = calib_data['intrinsic'].get('k3', 0.0)
                        camera_params[CameraParamIndex.K4] = calib_data['intrinsic'].get('k4', 0.0)
                        camera_params[CameraParamIndex.P1] = 0.0
                        camera_params[CameraParamIndex.P2] = 0.0
                    elif camera_type == CameraType.GENERAL_DISTORT:
                        camera_params[CameraParamIndex.K1] = calib_data['intrinsic'].get('k1', 0.0)
                        camera_params[CameraParamIndex.K2] = calib_data['intrinsic'].get('k2', 0.0)
                        camera_params[CameraParamIndex.K3] = calib_data['intrinsic'].get('k3', 0.0)
                        camera_params[CameraParamIndex.K4] = 0.0
                        camera_params[CameraParamIndex.P1] = calib_data['intrinsic'].get('p1', 0.0)
                        camera_params[CameraParamIndex.P2] = calib_data['intrinsic'].get('p2', 0.0)
                    elif camera_type == CameraType.PINHOLE:
                        camera_params[CameraParamIndex.K1:CameraParamIndex.P2+1] = 0.0
                    else:
                        raise ValueError(f"Unsupported camera type: {camera_type}")

                    # Set extrinsic parameters
                    camera_params[CameraParamIndex.X] = calib_data['extrinsic']['tx']
                    camera_params[CameraParamIndex.Y] = calib_data['extrinsic']['ty']
                    camera_params[CameraParamIndex.Z] = calib_data['extrinsic']['tz']
                    camera_params[CameraParamIndex.QX] = calib_data['extrinsic']['qx']
                    camera_params[CameraParamIndex.QY] = calib_data['extrinsic']['qy']
                    camera_params[CameraParamIndex.QZ] = calib_data['extrinsic']['qz']
                    camera_params[CameraParamIndex.QW] = calib_data['extrinsic']['qw']

                    # Store camera parameters
                    calibrations[camera_id] = camera_params

            
            # 2. Load ego states and labels
            with open(os.path.join(clip_dir, 'ego_states.json'), 'r') as f:
                ego_states = json.load(f)
            with open(os.path.join(clip_dir, 'obstacle_labels.json'), 'r') as f:
                obstacle_labels = json.load(f)
                
            # Verify data length
            assert len(ego_states) == len(obstacle_labels), \
                f"ego_states and labels should have same length but got {len(ego_states)} and {len(obstacle_labels)}"
            
            # Create samples with sliding window
            valid_sample_num = max(0, len(ego_states) - self.sequence_length + 1)
            for start_idx in range(valid_sample_num):
                sample = TrainingSample(calibrations)
                
                # Add frames to sample
                for i in range(start_idx, start_idx + self.sequence_length):
                    timestamp = "{:.6f}".format(ego_states[i]['timestamp'])
                    
                    # Collect image paths
                    img_paths = {}
                    for camera_id in self.camera_ids:
                        img_path = os.path.join(
                            clip_dir, 
                            camera_id.name.lower(),
                            f'{timestamp}.png'
                        )
                        assert os.path.exists(img_path), f"Image not found: {img_path}"
                        img_paths[camera_id] = img_path
                    
                    # Collect Ego State
                    ego_state = torch.zeros(EgoStateIndex.END_OF_INDEX)
                    ego_state[EgoStateIndex.X] = ego_states[i].get('x', 0.0)
                    ego_state[EgoStateIndex.Y] = ego_states[i].get('y', 0.0)
                    ego_state[EgoStateIndex.YAW] = ego_states[i].get('yaw', 0.0)
                    ego_state[EgoStateIndex.PITCH_CORRECTION] = ego_states[i].get('pitch_correction', 0.0)
                    ego_state[EgoStateIndex.VX] = ego_states[i].get('vx', 0.0)
                    ego_state[EgoStateIndex.VY] = ego_states[i].get('vy', 0.0)
                    ego_state[EgoStateIndex.AX] = ego_states[i].get('ax', 0.0)
                    ego_state[EgoStateIndex.AY] = ego_states[i].get('ay', 0.0)
                    ego_state[EgoStateIndex.YAW_RATE] = ego_states[i].get('yaw_rate', 0.0)
                    sample.add_frame(img_paths, ego_state)
                    
                    # Add trajectory if it's the last frame
                    if i == start_idx + self.sequence_length - 1:
                        trajs = []
                        for obj_data in obstacle_labels[i]['obstacles']:
                            traj = torch.zeros(TrajParamIndex.END_OF_INDEX)
                            
                            traj[TrajParamIndex.X] = obj_data.get('x', 0.0)
                            traj[TrajParamIndex.Y] = obj_data.get('y', 0.0)
                            traj[TrajParamIndex.Z] = obj_data.get('z', 0.0)
                            traj[TrajParamIndex.VX] = obj_data.get('rel_vx', 0.0)
                            traj[TrajParamIndex.VY] = obj_data.get('rel_vy', 0.0)
                            traj[TrajParamIndex.AX] = obj_data.get('rel_ax', 0.0)
                            traj[TrajParamIndex.AY] = obj_data.get('rel_ay', 0.0)
                            traj[TrajParamIndex.YAW] = obj_data.get('yaw', 0.0)
                            traj[TrajParamIndex.LENGTH] = obj_data.get('length', 0.0)
                            traj[TrajParamIndex.WIDTH] = obj_data.get('width', 0.0)
                            traj[TrajParamIndex.HEIGHT] = obj_data.get('height', 0.0)
                            traj[TrajParamIndex.HAS_OBJECT] = 1.0
                            traj[TrajParamIndex.STATIC] = obj_data.get('static', 0.0)
                            traj[TrajParamIndex.OCCLUDED] = obj_data.get('occluded', 0.0)
                            obj_type_str = obj_data.get('type', 'UNKNOWN')
                            traj[TrajParamIndex.OBJECT_TYPE] = float(ObjectType[obj_type_str])
                            
                            trajs.append(traj)
                        assert len(trajs) <= MAX_TRAJ_NB, f"Number of trajectories exceeds MAX_TRAJ_NB: {len(trajs)}"
                        sample.trajs = torch.stack(trajs)
                
                samples.append(sample)
            
            print(f"Clip {clip_dir} has {valid_sample_num} valid samples")
            
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[Dict, List, torch.Tensor]]:
        sample = self.samples[idx]
        T = len(sample.ego_states)
        
        
        ret = {
            'images': {},  # Dict[camera_id -> Tensor[T, C, H, W]]
            'calibrations': sample.calibrations,  # Dict[camera_id -> Tensor[CameraParamIndex.END_OF_INDEX]]
            'ego_states': torch.stack(sample.ego_states),  # Tensor[T, EgoStateIndex.END_OF_INDEX]
            'trajs': sample.trajs  # Tensor[-1, TrajParamIndex.END_OF_INDEX]
        }
        
        # Load images for each camera
        for camera_id in self.camera_ids:
            images = []
            for frame_paths in sample.image_paths:
                img = Image.open(frame_paths[camera_id])
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            ret['images'][camera_id] = torch.stack(images)
            
        return ret

    
def custom_collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for DataLoader."""
    B = len(batch)
    T = len(batch[0]['ego_states'])
    
    collated = {
        'images': {},      # Dict[camera_id -> Tensor[B, T, C, H, W]]
        'calibrations': {},  # Dict[camera_id -> Tensor[B, CameraParamIndex.END_OF_INDEX]]
        'trajs': torch.zeros(B, MAX_TRAJ_NB, TrajParamIndex.END_OF_INDEX),  # Tensor[B, MAX_TRAJ_NB, TrajParamIndex.END_OF_INDEX]   
        'ego_states': torch.stack([b['ego_states'] for b in batch])  # Tensor[B, T, EgoStateIndex.END_OF_INDEX]
    }
    
    # Collate images
    for camera_id in batch[0]['images'].keys():
        collated['images'][camera_id] = torch.stack([b['images'][camera_id] for b in batch])
    
    # Collate calibrations
    for camera_id in batch[0]['calibrations'].keys():
        collated['calibrations'][camera_id] = torch.stack([b['calibrations'][camera_id] for b in batch])

    # Collate trajectories
    for b_idx, b in enumerate(batch):
        collated['trajs'][b_idx, :len(b['trajs'])] = b['trajs']
            
    return collated


if __name__ == '__main__':
    # Test configuration
    clip_dirs = [
        '/home/xinjian/Code/VAutoLabelerCore/labeling_info/1_20231219T122348_pdb-l4e-c0011_0_0to8',
        '/home/xinjian/Code/VAutoLabelerCore/labeling_info/3_20240117T084829_pdb-l4e-b0005_10_792to812',
        '/home/xinjian/Code/VAutoLabelerCore/labeling_info/4_20240223T161731_pdb-l4e-b0001_4_197to207',
        '/home/xinjian/Code/VAutoLabelerCore/labeling_info/5_20240223T161731_pdb-l4e-b0001_4_159to169'
    ]

    camera_ids = [
        SourceCameraId.FRONT_CENTER_CAMERA,
        SourceCameraId.FRONT_LEFT_CAMERA,
        SourceCameraId.FRONT_RIGHT_CAMERA,
        SourceCameraId.SIDE_LEFT_CAMERA,
        SourceCameraId.SIDE_RIGHT_CAMERA,
        SourceCameraId.REAR_LEFT_CAMERA,
        SourceCameraId.REAR_RIGHT_CAMERA
    ]

    print("\n=== Testing Dataset Initialization ===")
    dataset = MultiFrameDataset(
        clip_dirs=clip_dirs,
        camera_ids=camera_ids,
        sequence_length=10
    )

    print("\n=== Testing Data Loading ===")
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=custom_collate_fn
    )

    print("\n=== Testing Batch Processing ===")
    for batch in dataloader:
        print("\nBatch contents:")
        print(f"Number of cameras: {len(batch['images'])}")
        
        print(batch.keys())
        
        print("\nImage shapes:")
        for camera_id, images in batch['images'].items():
            print(f"Camera {camera_id.name}: {images.shape}")
        
        print("\nEgo state info:")
        print(f"Shape of ego states tensor: {batch['ego_states'].shape}")
        
        print("\nTrajectory info:")
        print(f"Shape of trajectories tensor: {batch['trajs'].shape}")
        
        # Count valid objects (those with HAS_OBJECT=1)
        valid_traj_counts = (batch['trajs'][:, :, TrajParamIndex.HAS_OBJECT] > 0.5).sum(dim=1)
        print(f"Valid trajectories per sample: {valid_traj_counts}")
        
        # Print info about the first valid trajectory in the first sample
        if valid_traj_counts[0] > 0:
            # Find first valid trajectory
            first_valid_idx = torch.nonzero(batch['trajs'][0, :, TrajParamIndex.HAS_OBJECT] > 0.5)[0].item()
            first_traj = batch['trajs'][0, first_valid_idx]
            
            print(f"\nFirst valid trajectory details:")
            print(f"  Position: ({first_traj[TrajParamIndex.X]:.2f}, {first_traj[TrajParamIndex.Y]:.2f}, {first_traj[TrajParamIndex.Z]:.2f})")
            print(f"  Velocity: ({first_traj[TrajParamIndex.VX]:.2f}, {first_traj[TrajParamIndex.VY]:.2f})")
            print(f"  Dimensions: {first_traj[TrajParamIndex.LENGTH]:.2f} x {first_traj[TrajParamIndex.WIDTH]:.2f} x {first_traj[TrajParamIndex.HEIGHT]:.2f}")
            print(f"  Yaw: {first_traj[TrajParamIndex.YAW]:.2f}")
            print(f"  Object type: {ObjectType(int(first_traj[TrajParamIndex.OBJECT_TYPE].item()))}")
        
        print("\nCalibration info:")
        for camera_id, calib in batch['calibrations'].items():
            print(f"\nCamera {camera_id.name}:")
            print(f"  Camera parameters shape: {calib.shape}")
            print(f"  Camera type: {CameraType(int(calib[0, CameraParamIndex.CAMERA_TYPE].item()))}")
        
        # Only process one batch
        break

    print("\n=== Test Completed Successfully ===")
