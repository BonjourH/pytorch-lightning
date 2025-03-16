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
    EgoStateIndex, tensor_to_object_type
)
from configs.config import DataConfig
import numpy as np
from utils.math_utils import quaternion2RotationMatix


MAX_TRAJ_NB = 100

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
        
    @staticmethod
    def read_seqeuntial_images_to_tensor(img_paths: List[Dict[SourceCameraId, str]], device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Read images from image paths.
        
        Args:
            img_paths: List[List[Dict[SourceCameraId, str]]]
            
        Returns:
            Dict[SourceCameraId, torch.Tensor]
            camera_id -> torch.Tensor[B, T, C, H, W]
        """
        imgs = {cam_id: [] for cam_id in img_paths[0][0].keys()}
        B, T = len(img_paths), len(img_paths[0])
        for b_idx in range(B):
            for t_idx in range(T):
                for cam_id, img_path in img_paths[b_idx][t_idx].items():
                    img = Image.open(img_path)
                    img_np = np.array(img)
                    if len(img_np.shape) == 3:
                        img_np = img_np.transpose(2, 0, 1)
                    img = torch.from_numpy(img_np).float().to(device)
                    imgs[cam_id].append(img)
        for cam_id in imgs.keys():
            C, H, W = imgs[cam_id][0].shape
            imgs[cam_id] = torch.stack(imgs[cam_id])
            imgs[cam_id] = imgs[cam_id].view(B, T, C, H, W).permute(0, 1, 3, 4, 2) # [B, T, H, W, C]
        return imgs
    
class MultiFrameDataset(Dataset):
    """Dataset for multi-frame multi-camera perception."""
    def __init__(self, clip_dirs: List[str], config: DataConfig = None):
        assert config is not None, "config must be provided"
        self.config = config
        self.clip_dirs = clip_dirs
        self.samples = self._build_samples()
        print(f"Total {len(self.samples)} samples")

    def _build_samples(self) -> List[TrainingSample]:
        """Build list of training samples from clips."""
        samples = []
        
        for clip_dir in self.clip_dirs:
            # 1. Load calibrations
            calibrations: Dict[SourceCameraId, torch.Tensor] = {}
            for camera_id in self.config.camera_ids:
                camera_id = SourceCameraId(camera_id)
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
                    # from ego frame to camera frame
                    qx = float(calib_data['extrinsic']['qx'])
                    qy = float(calib_data['extrinsic']['qy'])
                    qz = float(calib_data['extrinsic']['qz'])
                    qw = float(calib_data['extrinsic']['qw'])
                    rot_ego_to_camera = torch.tensor([
                        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]
                    ])
                    t_ego_to_camera = torch.tensor([
                        float(calib_data['extrinsic']['tx']),
                        float(calib_data['extrinsic']['ty']),
                        float(calib_data['extrinsic']['tz'])
                    ])
                    rot_camera_to_ego = rot_ego_to_camera.transpose(0, 1)
                    t_camera_to_ego = -rot_camera_to_ego @ t_ego_to_camera
                    
                    camera_params[CameraParamIndex.R_EGO_TO_CAMERA_11:CameraParamIndex.R_EGO_TO_CAMERA_33+1] = rot_ego_to_camera.reshape(-1)
                    camera_params[CameraParamIndex.T_EGO_TO_CAMERA_X:CameraParamIndex.T_EGO_TO_CAMERA_Z+1] = t_ego_to_camera.reshape(-1)
                    camera_params[CameraParamIndex.R_CAMERA_TO_EGO_11:CameraParamIndex.R_CAMERA_TO_EGO_33+1] = rot_camera_to_ego.reshape(-1)
                    camera_params[CameraParamIndex.T_CAMERA_TO_EGO_X:CameraParamIndex.T_CAMERA_TO_EGO_Z+1] = t_camera_to_ego.reshape(-1)
                    
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
            valid_sample_num = max(0, len(ego_states) - self.config.sequence_length + 1)
            for start_idx in range(valid_sample_num):
                sample = TrainingSample(calibrations)
                
                # Get state for the last frame
                last_ego_state = ego_states[start_idx + self.config.sequence_length - 1]
                time_ref = float(last_ego_state['timestamp'])
                
                # calculate transformation from ego frame to global frame
                qw, qx, qy, qz = last_ego_state['qw'], last_ego_state['qx'], last_ego_state['qy'], last_ego_state['qz']
                tx, ty, tz = last_ego_state['x'], last_ego_state['y'], last_ego_state['z']
                rot_ego_to_global = quaternion2RotationMatix(qw, qx, qy, qz)
                t_ego_to_global = torch.tensor([tx, ty, tz])
                
                
                # x_ref, y_ref = last_ego_state['x'], last_ego_state['y']
                # qw, qx, qy, qz = last_ego_state['qw'], last_ego_state['qx'], last_ego_state['qy'], last_ego_state['qz']
                # yaw_ref = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                
                # Add frames to sample
                for i in range(start_idx, start_idx + self.config.sequence_length):
                    timestamp = "{:.6f}".format(ego_states[i]['timestamp'])
                    
                    # Collect image paths
                    img_paths = {}
                    for camera_id in self.config.camera_ids:
                        camera_id = SourceCameraId(camera_id)
                        img_path = os.path.join(clip_dir, camera_id.name.lower(), f'{timestamp}.png')
                        assert os.path.exists(img_path), f"Image not found: {img_path}"
                        img_paths[camera_id] = img_path
                    
                    # Collect Ego State
                    # ego_state = torch.zeros(EgoStateIndex.END_OF_INDEX, dtype=torch.float64)  # 使用 float64 保证精度
                    ego_state = torch.zeros(EgoStateIndex.END_OF_INDEX)
                    # 确保时间戳保留小数精度
                    timestamp_float = float(ego_states[i]['timestamp'])
                    ego_state[EgoStateIndex.TIMESTAMP] = timestamp_float - time_ref
                    qw, qx, qy, qz = ego_states[i]['qw'], ego_states[i]['qx'], ego_states[i]['qy'], ego_states[i]['qz']
                    rot_prev_ego_to_global = quaternion2RotationMatix(qw, qx, qy, qz)
                    t_prev_ego_to_global = torch.tensor([ego_states[i]['x'], ego_states[i]['y'], ego_states[i]['z']])
                    t_prev_ego_to_global = t_prev_ego_to_global.to(rot_prev_ego_to_global.dtype)
                    
                    rot_ego_to_prev_ego = rot_prev_ego_to_global.transpose(0, 1) @ rot_ego_to_global
                    t_ego_to_prev_ego = rot_prev_ego_to_global.transpose(0, 1) @ (t_ego_to_global - t_prev_ego_to_global)
                    
                    # calculate yaw, x, y from ego frame to previous ego frame
                    yaw_rgo_to_prev_ego = np.arctan2(rot_ego_to_prev_ego[1, 0], rot_ego_to_prev_ego[0, 0])
                    ego_state[EgoStateIndex.YAW] = yaw_rgo_to_prev_ego
                    ego_state[EgoStateIndex.X] = t_ego_to_prev_ego[0]
                    ego_state[EgoStateIndex.Y] = t_ego_to_prev_ego[1]

                    # yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                    # ego_state[EgoStateIndex.YAW] = yaw_ref - yaw
                    
                    # x_diff = x_ref - ego_states[i]['x']
                    # y_diff = y_ref - ego_states[i]['y']
                    # cos_val, sin_val = np.cos(yaw), np.sin(yaw)
                    # ego_state[EgoStateIndex.X] = x_diff * cos_val + y_diff * sin_val
                    # ego_state[EgoStateIndex.Y] = -x_diff * sin_val + y_diff * cos_val
                    
                    # NOTE: the origin ego state is in first ego frame, but here we want to change the 
                    # relative ego state to the last ego frame. The question is: we have (x1, y1, yaw1) in ego1 frame1,
                    # and (x2, y2, yaw2) in ego2 frame2, how to convert (x1, y1, yaw1) to ego2 frame?
                    # the answer is:
                    # frame1 to global frame is (R1, t1), frame2 to global frame is (R2, t2),
                    # global frame to frame2 is (R2^T, -R2^T * t2),
                    # then frame1 to frame2 is (R2^T * R1, R2^T * (t1 - t2)) => (yaw1 - yaw2, xx, xx)
                    # here frame1 is the last ego frame, frame2 is the current ego frame we set the variable
                    
                    ego_state[EgoStateIndex.PITCH_CORRECTION] = ego_states[i]['pitch']
                    # ego_state[EgoStateIndex.VX] = ego_states[i].get('speed', 0.0)
                    # ego_state[EgoStateIndex.VY] = ego_states[i].get('vy', 0.0)
                    # ego_state[EgoStateIndex.AX] = ego_states[i].get('acceleration', 0.0)
                    # ego_state[EgoStateIndex.AY] = ego_states[i].get('ay', 0.0)
                    # ego_state[EgoStateIndex.YAW_RATE] = ego_states[i].get('yaw_rate', 0.0)
                    
                    sample.add_frame(img_paths, ego_state)
                    
                    # Add trajectory if it's the last frame
                    if i == start_idx + self.config.sequence_length - 1:
                        trajs = []
                        for obj_data in obstacle_labels[i]['obstacles']:
                            traj = torch.zeros(TrajParamIndex.END_OF_INDEX)
                            # motion parameters
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
                            # object attributes
                            traj[TrajParamIndex.HAS_OBJECT] = 1.0
                            traj[TrajParamIndex.STATIC] = obj_data.get('static', 0.0)
                            traj[TrajParamIndex.OCCLUDED] = obj_data.get('occluded', 0.0)
                            # object type
                            traj[TrajParamIndex.object_type_to_index(obj_data.get('type', 'UNKNOWN'))] = 1.0
                            
                            trajs.append(traj)
                        assert len(trajs) <= MAX_TRAJ_NB, f"Number of trajectories exceeds MAX_TRAJ_NB: {len(trajs)}"
                        sample.trajs = torch.stack(trajs)
                
                samples.append(sample)
            
            print(f"Clip {clip_dir} has {valid_sample_num} valid samples")
            
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[Dict, List, torch.Tensor]]:
        # load images from image_paths and do preprocessing
        sample = self.samples[idx]
        T = len(sample.ego_states)
        
        ret = {
            'image_paths': sample.image_paths, # List[Dict[SourceCameraId, str]]
            'images': {},  # Dict[str -> Tensor[T, N_cams, C, H, W]]
            'calibrations': sample.calibrations,  # Dict[camera_id -> Tensor[CameraParamIndex.END_OF_INDEX]]
            'ego_states': torch.stack(sample.ego_states),  # Tensor[T, EgoStateIndex.END_OF_INDEX]
            'trajs': sample.trajs  # Tensor[-1, TrajParamIndex.END_OF_INDEX]
        }
        
        # Load images for each camera group
        for camera_group in self.config.camera_groups:
            transform = transforms.Compose([    
                transforms.Resize(camera_group.image_size),
                transforms.ToTensor(),
                transforms.Normalize(camera_group.normalize_mean, camera_group.normalize_std)
            ])
            imgs = []
            for img_paths in sample.image_paths:
                frame_imgs = []
                for camera_id in camera_group.camera_group:
                    img = Image.open(img_paths[camera_id])
                    img = transform(img)
                    frame_imgs.append(img)
                one_frame_img = torch.stack(frame_imgs) # [N_cams, C, H, W]
                imgs.append(one_frame_img)
            ret['images'][camera_group.name] = torch.stack(imgs) # [T, N_cams, C, H, W]
        return ret

CAMERA_ID_LIST = [
    SourceCameraId.FRONT_LEFT_CAMERA,
    SourceCameraId.FRONT_RIGHT_CAMERA,
    SourceCameraId.FRONT_CENTER_CAMERA,
    SourceCameraId.SIDE_LEFT_CAMERA,
    SourceCameraId.SIDE_RIGHT_CAMERA,
    SourceCameraId.REAR_LEFT_CAMERA,
    SourceCameraId.REAR_RIGHT_CAMERA,
]

CAMERA_ID_TO_INDEX = {camera_id: i for i, camera_id in enumerate(CAMERA_ID_LIST)}
    
def custom_collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for DataLoader."""
    B = len(batch)
    T = len(batch[0]['ego_states'])
    
    collated = {
        'image_paths': [b['image_paths'] for b in batch],  # List[Dict[SourceCameraId, str]]
        'images': {},      # Dict[str -> Tensor[B, T, N_cams, C, H, W]]
        'camera_ids': CAMERA_ID_LIST,
        'calibrations': torch.zeros(B, 7, CameraParamIndex.END_OF_INDEX),
        'valid_traj_nb': torch.zeros(B),  # Tensor[B]
        'trajs': torch.zeros(B, MAX_TRAJ_NB, TrajParamIndex.END_OF_INDEX),  # Tensor[B, MAX_TRAJ_NB, TrajParamIndex.END_OF_INDEX]   
        'ego_states': torch.stack([b['ego_states'] for b in batch])  # Tensor[B, T, EgoStateIndex.END_OF_INDEX]
    }
    
    # Collate images
    for camera_group_name in batch[0]['images'].keys():
        collated['images'][camera_group_name] = torch.stack([b['images'][camera_group_name] for b in batch])
    
    # Collate calibrations
    for b_idx, b in enumerate(batch):
        for camera_id in b['calibrations'].keys():
            collated['calibrations'][b_idx, CAMERA_ID_TO_INDEX[camera_id]] = b['calibrations'][camera_id]

    # Collate trajectories
    for b_idx, b in enumerate(batch):
        collated['trajs'][b_idx, :len(b['trajs'])] = b['trajs']
        collated['valid_traj_nb'][b_idx] = len(b['trajs'])
            
    return collated


if __name__ == '__main__':
    # Test configuration
    clip_dirs = [
        '/home/xinjian/Code/VAutoLabelerCore/labeling_info/1_20231219T122348_pdb-l4e-c0011_0_0to8'
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
        config=DataConfig()
    )

    print("\n=== Testing Data Loading ===")
    dataloader = DataLoader(
        dataset,
        batch_size=5,
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
        for camera_group_name, images in batch['images'].items():
            print(f"Camera group {camera_group_name}: {images.shape} (batch_size, T, N_cams, C, H, W)")
        
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
            print(f"  Object type: {tensor_to_object_type(first_traj)}")
        
        print("\nCalibration info:")
        calib =  batch['calibrations'][0]
        print(f"  Camera parameters shape: {calib.shape}")
        
        # Only process one batch
        break

    print("\n=== Test Completed Successfully ===")
