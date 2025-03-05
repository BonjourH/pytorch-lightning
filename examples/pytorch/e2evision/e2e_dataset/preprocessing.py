import torch
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
from base import SourceCameraId, CameraType, CameraParamIndex, EgoStateIndex

class ImagePreprocessor:
    """Image preprocessing tools."""
    
    def __init__(self,
                 target_size: Tuple[int, int] = (224, 224),
                 normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        self.target_size = target_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
    
    def __call__(self, image: Union[str, np.ndarray]) -> torch.Tensor:
        """Process image.
        
        Args:
            image: Image path or numpy array
            
        Returns:
            Processed image tensor [C, H, W]
        """
        # Load image if path
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, self.target_size)
        
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        image = (image - np.array(self.normalize_mean)) / np.array(self.normalize_std)
        
        # Convert to tensor and transpose
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image

class CalibrationPreprocessor:
    """Camera calibration preprocessing tools."""
    
    def __init__(self):
        pass
    
    def __call__(self, calib_data: Dict) -> torch.Tensor:
        """Process calibration data.
        
        Args:
            calib_data: Dictionary containing calibration parameters
            
        Returns:
            Calibration tensor [CameraParamIndex.END_OF_INDEX]
        """
        calib_tensor = torch.zeros(CameraParamIndex.END_OF_INDEX)
        
        # Fill calibration parameters
        calib_tensor[CameraParamIndex.CAMERA_ID] = calib_data.get('camera_id', 0)
        calib_tensor[CameraParamIndex.CAMERA_TYPE] = calib_data.get('camera_type', CameraType.PINHOLE)
        calib_tensor[CameraParamIndex.IMAGE_WIDTH] = calib_data.get('image_width', 0)
        calib_tensor[CameraParamIndex.IMAGE_HEIGHT] = calib_data.get('image_height', 0)
        calib_tensor[CameraParamIndex.FX] = calib_data.get('fx', 0.0)
        calib_tensor[CameraParamIndex.FY] = calib_data.get('fy', 0.0)
        calib_tensor[CameraParamIndex.CX] = calib_data.get('cx', 0.0)
        calib_tensor[CameraParamIndex.CY] = calib_data.get('cy', 0.0)
        
        # Distortion parameters
        calib_tensor[CameraParamIndex.K1] = calib_data.get('k1', 0.0)
        calib_tensor[CameraParamIndex.K2] = calib_data.get('k2', 0.0)
        calib_tensor[CameraParamIndex.K3] = calib_data.get('k3', 0.0)
        calib_tensor[CameraParamIndex.K4] = calib_data.get('k4', 0.0)
        calib_tensor[CameraParamIndex.P1] = calib_data.get('p1', 0.0)
        calib_tensor[CameraParamIndex.P2] = calib_data.get('p2', 0.0)
        
        # Extrinsic parameters
        calib_tensor[CameraParamIndex.X] = calib_data.get('x', 0.0)
        calib_tensor[CameraParamIndex.Y] = calib_data.get('y', 0.0)
        calib_tensor[CameraParamIndex.Z] = calib_data.get('z', 0.0)
        calib_tensor[CameraParamIndex.QX] = calib_data.get('qx', 0.0)
        calib_tensor[CameraParamIndex.QY] = calib_data.get('qy', 0.0)
        calib_tensor[CameraParamIndex.QZ] = calib_data.get('qz', 0.0)
        calib_tensor[CameraParamIndex.QW] = calib_data.get('qw', 1.0)
        
        return calib_tensor

class EgoStatePreprocessor:
    """Ego vehicle state preprocessing tools."""
    
    def __init__(self):
        pass
    
    def __call__(self, ego_data: Dict) -> torch.Tensor:
        """Process ego state data.
        
        Args:
            ego_data: Dictionary containing ego state parameters
            
        Returns:
            Ego state tensor [EgoStateIndex.END_OF_INDEX]
        """
        ego_tensor = torch.zeros(EgoStateIndex.END_OF_INDEX)
        
        # Fill ego state parameters
        ego_tensor[EgoStateIndex.X] = ego_data.get('x', 0.0)
        ego_tensor[EgoStateIndex.Y] = ego_data.get('y', 0.0)
        ego_tensor[EgoStateIndex.YAW] = ego_data.get('yaw', 0.0)
        ego_tensor[EgoStateIndex.PITCH_CORRECTION] = ego_data.get('pitch_correction', 0.0)
        ego_tensor[EgoStateIndex.VX] = ego_data.get('vx', 0.0)
        ego_tensor[EgoStateIndex.VY] = ego_data.get('vy', 0.0)
        ego_tensor[EgoStateIndex.AX] = ego_data.get('ax', 0.0)
        ego_tensor[EgoStateIndex.AY] = ego_data.get('ay', 0.0)
        ego_tensor[EgoStateIndex.YAW_RATE] = ego_data.get('yaw_rate', 0.0)
        
        return ego_tensor

class DataPreprocessor:
    """Main data preprocessing class."""
    
    def __init__(self,
                 image_size: Tuple[int, int] = (224, 224),
                 normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        self.image_processor = ImagePreprocessor(
            target_size=image_size,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std
        )
        self.calib_processor = CalibrationPreprocessor()
        self.ego_processor = EgoStatePreprocessor()
    
    def process_clip(self, clip_dir: Union[str, Path]) -> Dict:
        """Process a single clip directory.
        
        Args:
            clip_dir: Path to clip directory
            
        Returns:
            Dict containing processed data
        """
        clip_dir = Path(clip_dir)
        
        # Load metadata
        with open(clip_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Process images
        images = {}
        for camera_id in metadata['cameras']:
            camera_images = []
            for frame_idx in range(metadata['num_frames']):
                image_path = clip_dir / 'images' / f'{camera_id}_{frame_idx:06d}.jpg'
                if image_path.exists():
                    processed_image = self.image_processor(str(image_path))
                    camera_images.append(processed_image)
            if camera_images:
                images[SourceCameraId(int(camera_id))] = torch.stack(camera_images)
        
        # Process calibrations
        calibrations = {}
        for camera_id, calib_data in metadata['calibrations'].items():
            calibrations[SourceCameraId(int(camera_id))] = self.calib_processor(calib_data)
        
        # Process ego states
        ego_states = []
        for frame_idx in range(metadata['num_frames']):
            ego_path = clip_dir / 'ego' / f'ego_{frame_idx:06d}.json'
            if ego_path.exists():
                with open(ego_path, 'r') as f:
                    ego_data = json.load(f)
                ego_states.append(self.ego_processor(ego_data))
        ego_states = torch.stack(ego_states) if ego_states else torch.zeros(0, EgoStateIndex.END_OF_INDEX)
        
        return {
            'images': images,
            'calibrations': calibrations,
            'ego_states': ego_states,
            'metadata': metadata
        } 