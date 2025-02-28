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
    CameraCalibration, SourceCameraId, CameraType, 
    ObjectType, ObstacleTrajectory, Point3DAccMotion
)
import numpy as np


class TrainingSample:
    """Container for multi-frame training data."""
    def __init__(self, calibrations: Dict[SourceCameraId, CameraCalibration]):
        self.calibrations = calibrations
        self.image_paths: List[Dict[SourceCameraId, str]] = []
        self.ego_states: List[Dict] = []
        self.objects: List[List[ObstacleTrajectory]] = []
    
    def add_frame(self, 
                 image_paths: Dict[SourceCameraId, str],
                 ego_state: Dict,
                 objects: List[ObstacleTrajectory]):
        self.image_paths.append(image_paths)
        self.ego_states.append(ego_state)
        self.objects.append(objects)

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
            calibrations = {}
            for camera_id in self.camera_ids:
                calib_file = os.path.join(clip_dir, 'calib_json', f'{camera_id.name.lower()}.json')
                with open(calib_file, 'r') as f:
                    calib_data = json.load(f)
                    
                    # Convert intrinsic parameters to matrix
                    intrinsic = torch.zeros(3, 3)
                    intrinsic[0, 0] = calib_data['intrinsic']['fx']
                    intrinsic[1, 1] = calib_data['intrinsic']['fy']
                    intrinsic[0, 2] = calib_data['intrinsic']['cx']
                    intrinsic[1, 2] = calib_data['intrinsic']['cy']
                    intrinsic[2, 2] = 1.0

                    # Convert extrinsic parameters
                    qw = calib_data['extrinsic']['qw']
                    qx = calib_data['extrinsic']['qx']
                    qy = calib_data['extrinsic']['qy']
                    qz = calib_data['extrinsic']['qz']
                    tx = calib_data['extrinsic']['tx']
                    ty = calib_data['extrinsic']['ty']
                    tz = calib_data['extrinsic']['tz']
                    
                    # Convert quaternion to rotation matrix
                    extrinsic = torch.zeros(4, 4)
                    extrinsic[3, 3] = 1.0
                    
                    # Quaternion to rotation matrix
                    extrinsic[0, 0] = 1 - 2*qy*qy - 2*qz*qz
                    extrinsic[0, 1] = 2*qx*qy - 2*qz*qw
                    extrinsic[0, 2] = 2*qx*qz + 2*qy*qw
                    extrinsic[1, 0] = 2*qx*qy + 2*qz*qw
                    extrinsic[1, 1] = 1 - 2*qx*qx - 2*qz*qz
                    extrinsic[1, 2] = 2*qy*qz - 2*qx*qw
                    extrinsic[2, 0] = 2*qx*qz - 2*qy*qw
                    extrinsic[2, 1] = 2*qy*qz + 2*qx*qw
                    extrinsic[2, 2] = 1 - 2*qx*qx - 2*qy*qy
                    
                    # Add translation
                    extrinsic[0, 3] = tx
                    extrinsic[1, 3] = ty
                    extrinsic[2, 3] = tz

                    # Get distortion coefficients based on camera type
                    if calib_data['type'] == 'FISHEYE':
                        distortion = torch.tensor([
                            calib_data['intrinsic']['k1'],
                            calib_data['intrinsic']['k2'],
                            calib_data['intrinsic']['k3'],
                            calib_data['intrinsic']['k4']
                        ])
                    elif calib_data['type'] == 'GENERAL_DISTORT':
                        distortion = torch.tensor([
                            calib_data['intrinsic']['k1'],
                            calib_data['intrinsic']['k2'],
                            calib_data['intrinsic']['p1'],
                            calib_data['intrinsic']['p2'],
                            calib_data['intrinsic']['k3']
                        ])
                    else:
                        distortion = None

                    calibrations[camera_id] = CameraCalibration(
                        camera_id=camera_id,
                        camera_type=CameraType[calib_data['type']],
                        intrinsic=intrinsic,
                        extrinsic=extrinsic,
                        distortion=distortion
                    )
            
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
                    
                    # Convert objects to Trajectory instances
                    objects = []
                    for obj_data in obstacle_labels[i]['obstacles']:
                        motion = Point3DAccMotion(
                            x=obj_data['x'],
                            y=obj_data['y'],
                            z=obj_data['z'],
                            vx=obj_data.get('vx', 0.0),
                            vy=obj_data.get('vy', 0.0),
                            vz=0.0,  # Assume no vertical velocity
                            ax=0.0,  # Initialize with zero acceleration
                            ay=0.0,
                            az=0.0
                        )
                        
                        obj = ObstacleTrajectory(
                            id=int(obj_data['id']),
                            t0=ego_states[i]['timestamp'],
                            motion=motion,
                            yaw=obj_data['yaw'],
                            length=obj_data['length'],
                            width=obj_data['width'],
                            height=obj_data['height'],
                            object_type=ObjectType[obj_data['type']],
                            valid=True
                        )
                        objects.append(obj)
                    
                    sample.add_frame(img_paths, ego_states[i], objects)
                
                samples.append(sample)
            
            print(f"Clip {clip_dir} has {valid_sample_num} valid samples")
            
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[Dict, List, torch.Tensor]]:
        sample = self.samples[idx]
        
        ret = {
            'images': {},  # Dict[camera_id -> Tensor[sequence_length, C, H, W]]
            'ego_states': [],  # List[Dict] of ego poses
            'objects_data': [],  # List[Dict] containing object tensors
            'calibrations': {}  # Dict[camera_id -> Dict] of calibration tensors
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

        # Add ego states
        ret['ego_states'] = sample.ego_states

        # Convert Trajectory objects to tensor format
        for frame_objects in sample.objects:
            frame_data = {
                'ids': [],
                'types': [],
                'positions': [],
                'dimensions': [],
                'yaws': [],
                'velocities': [],
                'accelerations': []  # Added acceleration data
            }
            
            for obj in frame_objects:
                frame_data['ids'].append(obj.id)
                frame_data['types'].append(int(obj.object_type))
                frame_data['positions'].append(obj.position)
                frame_data['dimensions'].append(obj.dimensions)
                frame_data['yaws'].append(obj.yaw)
                frame_data['velocities'].append(obj.velocity[:2])  # Only x-y velocity
                frame_data['accelerations'].append(obj.acceleration[:2])  # Only x-y acceleration
            
            # Convert lists to tensors
            if frame_objects:
                frame_data['types'] = torch.tensor(frame_data['types'])
                frame_data['positions'] = torch.tensor(np.stack(frame_data['positions']))
                frame_data['dimensions'] = torch.tensor(np.stack(frame_data['dimensions']))
                frame_data['yaws'] = torch.tensor(frame_data['yaws'])
                frame_data['velocities'] = torch.tensor(np.stack(frame_data['velocities']))
                frame_data['accelerations'] = torch.tensor(np.stack(frame_data['accelerations']))
            else:
                frame_data['types'] = torch.empty(0, dtype=torch.long)
                frame_data['positions'] = torch.empty(0, 3)
                frame_data['dimensions'] = torch.empty(0, 3)
                frame_data['yaws'] = torch.empty(0)
                frame_data['velocities'] = torch.empty(0, 2)
                frame_data['accelerations'] = torch.empty(0, 2)
                
            ret['objects_data'].append(frame_data)

        # Convert calibrations to tensor format
        for camera_id, calib in sample.calibrations.items():
            ret['calibrations'][camera_id] = {
                'camera_id': camera_id,
                'camera_type': calib.camera_type,
                'intrinsic': calib.intrinsic,
                'extrinsic': calib.extrinsic,
                'distortion': calib.distortion if calib.distortion is not None else torch.empty(0)
            }

        return ret
    
def custom_collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for DataLoader."""
    collated = {
        'images': {},      # Dict[camera_id -> Tensor[B, T, C, H, W]]
        'ego_states': [],  # List[B * List[Dict]]
        'objects_data': [], # List[B * List[Dict]]
        'calibrations': {} # Dict[camera_id -> Dict[str, Union[Tensor, CameraType]]]
    }
    
    # Collate images
    for camera_id in batch[0]['images'].keys():
        collated['images'][camera_id] = torch.stack([b['images'][camera_id] for b in batch])
    
    # Collate ego states (just extend the list)
    for b in batch:
        collated['ego_states'].extend(b['ego_states'])
    
    # Collate objects data (just extend the list)
    for b in batch:
        collated['objects_data'].extend(b['objects_data'])
    
    # Collate calibrations
    for camera_id in batch[0]['calibrations'].keys():
        calib_dict = {}
        first_calib = batch[0]['calibrations'][camera_id]
             
        # Handle non-tensor fields
        calib_dict['camera_id'] = first_calib['camera_id']
        calib_dict['camera_type'] = first_calib['camera_type']
        
        # Stack tensor fields
        calib_dict['intrinsic'] = torch.stack([b['calibrations'][camera_id]['intrinsic'] for b in batch])
        calib_dict['extrinsic'] = torch.stack([b['calibrations'][camera_id]['extrinsic'] for b in batch])
        
        # Handle distortion (might be empty)
        distortions = [b['calibrations'][camera_id]['distortion'] for b in batch]
        if distortions[0].numel() > 0:  # If not empty
            calib_dict['distortion'] = torch.stack(distortions)
        else:
            calib_dict['distortion'] = torch.empty(0)
            
        collated['calibrations'][camera_id] = calib_dict
    
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
        import pdb; pdb.set_trace()
        # batch['images'][SourceCameraId.FRONT_CENTER_CAMERA].shape => torch.Size([2, 10, 3, 416, 800]
        
        print("\nImage shapes:")
        for camera_id, images in batch['images'].items():
            print(f"Camera {camera_id.name}: {images.shape}")
        
        print("\nEgo state info:")
        print(f"Number of ego states: {len(batch['ego_states'])}")
        print(f"First ego state keys: {batch['ego_states'][0].keys()}")
        
        print("\nObject info:")
        print(f"Number of frames: {len(batch['objects_data'])}")
        if len(batch['objects_data']) > 0:
            frame_data = batch['objects_data'][0]
            print(f"First frame object data:")
            print(f"  Number of objects: {len(frame_data['ids'])}")
            if len(frame_data['ids']) > 0:
                print(f"  Types shape: {frame_data['types'].shape}")
                print(f"  Positions shape: {frame_data['positions'].shape}")
                print(f"  Dimensions shape: {frame_data['dimensions'].shape}")
                print(f"  Yaws shape: {frame_data['yaws'].shape}")
                print(f"  Velocities shape: {frame_data['velocities'].shape}")
        
        print("\nCalibration info:")
        for camera_id, calib in batch['calibrations'].items():
            print(f"\nCamera {camera_id.name}:")
            print(f"  Camera type: {calib['camera_type'].name}")
            print(f"  Intrinsic shape: {calib['intrinsic'].shape}")
            print(f"  Extrinsic shape: {calib['extrinsic'].shape}")
            if calib['distortion'].numel() > 0:  # Check if distortion tensor is not empty
                print(f"  Distortion shape: {calib['distortion'].shape}")
        
        # Only process one batch for testing
        break

    print("\n=== Test Completed Successfully ===")