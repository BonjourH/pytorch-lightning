import os
import json
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image

@dataclass
class CameraCalibration:
    """Camera calibration parameters."""
    intrinsic: torch.Tensor  # 3x3 camera intrinsic matrix
    extrinsic: torch.Tensor  # 4x4 camera extrinsic matrix
    distortion: Optional[torch.Tensor] = None  # distortion coefficients

class MultiCameraFrame:
    """Container for multi-camera frame data."""
    def __init__(self, timestamp: float, calibrations: Dict[str, CameraCalibration]):
        self.timestamp = timestamp
        self.calibrations = calibrations
        self.image_paths: Dict[str, str] = {}
        self.ego_state: Dict = {}
        self.objects_3d: List[Dict] = []

class MultiCameraDataset(Dataset):
    """Dataset for multi-camera perception."""
    def __init__(self, 
                 data_root: str,
                 camera_ids: List[str],
                 sequence_length: int = 10,
                 transform=None):
        self.data_root = data_root
        self.camera_ids = camera_ids
        self.sequence_length = sequence_length
        self.transform = transform or transforms.Compose([
            transforms.Resize((416, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        self.sequences = self._load_sequences()

    def _load_sequences(self) -> List[List[MultiCameraFrame]]:
        sequences = []
        # Load calibrations
        calibrations = {}
        for camera_id in self.camera_ids:
            calib_path = os.path.join(self.data_root, 'calibration', f'{camera_id}.json')
            with open(calib_path, 'r') as f:
                calib_data = json.load(f)
                calibrations[camera_id] = CameraCalibration(
                    intrinsic=torch.tensor(calib_data['intrinsic']),
                    extrinsic=torch.tensor(calib_data['extrinsic']),
                    distortion=torch.tensor(calib_data.get('distortion', []))
                )

        # Load frame data
        with open(os.path.join(self.data_root, 'frames.json'), 'r') as f:
            frames_data = json.load(f)

        # Group frames into sequences
        for i in range(0, len(frames_data) - self.sequence_length + 1):
            sequence = []
            for j in range(i, i + self.sequence_length):
                frame_data = frames_data[j]
                frame = MultiCameraFrame(
                    timestamp=frame_data['timestamp'],
                    calibrations=calibrations
                )
                frame.ego_state = frame_data['ego_state']
                frame.objects_3d = frame_data['objects_3d']
                
                # Load image paths
                for camera_id in self.camera_ids:
                    frame.image_paths[camera_id] = os.path.join(
                        self.data_root, 'images',
                        camera_id, f'{frame.timestamp:.6f}.jpg'
                    )
                sequence.append(frame)
            sequences.append(sequence)
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        
        # Prepare return dictionary
        ret = {
            'images': {},      # Dict[camera_id -> Tensor[sequence_length, C, H, W]]
            'ego_states': [],   # Tensor[sequence_length, state_dim]
            'objects': [],      # List[Tensor[num_objects, object_dim]]
            'calibrations': {}  # Dict[camera_id -> CameraCalibration]
        }

        # Load images for each camera
        for camera_id in self.camera_ids:
            images = []
            for frame in sequence:
                img = Image.open(frame.image_paths[camera_id])
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            ret['images'][camera_id] = torch.stack(images)

        # Load ego states and objects
        for frame in sequence:
            ret['ego_states'].append(torch.tensor(
                [frame.ego_state['x'], frame.ego_state['y'], 
                 frame.ego_state['heading']], dtype=torch.float32
            ))
            
            objects = torch.tensor([
                [obj['x'], obj['y'], obj['z'], 
                 obj['length'], obj['width'], obj['height'],
                 obj['heading']] for obj in frame.objects_3d
            ], dtype=torch.float32)
            ret['objects'].append(objects)

        ret['ego_states'] = torch.stack(ret['ego_states'])
        ret['calibrations'] = sequence[0].calibrations

        return ret