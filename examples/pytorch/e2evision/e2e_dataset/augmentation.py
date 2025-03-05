import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    # Image augmentations
    random_brightness: float = 0.2
    random_contrast: float = 0.2
    random_saturation: float = 0.2
    random_hue: float = 0.1
    random_flip: bool = True
    random_crop: bool = True
    crop_size: Tuple[int, int] = (224, 224)
    
    # Geometric augmentations
    random_scale: Tuple[float, float] = (0.8, 1.2)
    random_rotate: float = 10.0  # degrees
    
    # Noise augmentations
    gaussian_noise: float = 0.02
    random_erase_prob: float = 0.5
    
    # Weather augmentations
    random_weather: bool = True
    weather_types: List[str] = None
    
    def __post_init__(self):
        if self.weather_types is None:
            self.weather_types = ['rain', 'snow', 'fog']

class WeatherAugmentation:
    """Weather effect augmentation."""
    
    @staticmethod
    def add_rain(image: torch.Tensor, density: float = 0.1) -> torch.Tensor:
        """Add rain effect to image."""
        B, C, H, W = image.shape
        rain = torch.rand(B, C, H, W, device=image.device) > (1 - density)
        rain = rain.float() * 0.5
        return torch.clamp(image + rain, 0, 1)
    
    @staticmethod
    def add_snow(image: torch.Tensor, density: float = 0.1) -> torch.Tensor:
        """Add snow effect to image."""
        B, C, H, W = image.shape
        snow = torch.rand(B, C, H, W, device=image.device) > (1 - density)
        snow = snow.float()
        return torch.clamp(image + snow, 0, 1)
    
    @staticmethod
    def add_fog(image: torch.Tensor, density: float = 0.3) -> torch.Tensor:
        """Add fog effect to image."""
        fog = torch.ones_like(image) * density
        return torch.clamp(image * (1 - density) + fog, 0, 1)

class ImageAugmentation:
    """Image augmentation module."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.weather_aug = WeatherAugmentation()
        
        # Create transform pipeline
        self.transforms = []
        
        # Color transforms
        if config.random_brightness > 0:
            self.transforms.append(
                T.ColorJitter(brightness=config.random_brightness)
            )
        if config.random_contrast > 0:
            self.transforms.append(
                T.ColorJitter(contrast=config.random_contrast)
            )
        if config.random_saturation > 0:
            self.transforms.append(
                T.ColorJitter(saturation=config.random_saturation)
            )
        if config.random_hue > 0:
            self.transforms.append(
                T.ColorJitter(hue=config.random_hue)
            )
            
        # Geometric transforms
        if config.random_crop:
            self.transforms.append(
                T.RandomCrop(config.crop_size)
            )
        if config.random_flip:
            self.transforms.append(T.RandomHorizontalFlip())
            
        # Noise transforms
        if config.gaussian_noise > 0:
            self.transforms.append(
                T.Lambda(lambda x: x + torch.randn_like(x) * config.gaussian_noise)
            )
        if config.random_erase_prob > 0:
            self.transforms.append(
                T.RandomErasing(p=config.random_erase_prob)
            )
            
        self.transform = T.Compose(self.transforms)
    
    def __call__(self, images: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply augmentation to images.
        
        Args:
            images: Dict[camera_id -> Tensor[B, T, C, H, W]]
            
        Returns:
            Augmented images with same structure
        """
        augmented = {}
        
        for camera_id, image in images.items():
            # Apply standard transforms
            aug_image = self.transform(image)
            
            # Apply weather effects if enabled
            if self.config.random_weather and torch.rand(1) > 0.5:
                weather_type = np.random.choice(self.config.weather_types)
                if weather_type == 'rain':
                    aug_image = self.weather_aug.add_rain(aug_image)
                elif weather_type == 'snow':
                    aug_image = self.weather_aug.add_snow(aug_image)
                elif weather_type == 'fog':
                    aug_image = self.weather_aug.add_fog(aug_image)
                    
            augmented[camera_id] = aug_image
            
        return augmented

class GeometricAugmentation:
    """Geometric augmentation that preserves 3D consistency."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
    
    def __call__(self, 
                 images: Dict[str, torch.Tensor],
                 calibrations: Dict[str, torch.Tensor],
                 trajectories: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], 
                                                    Dict[str, torch.Tensor],
                                                    torch.Tensor]:
        """Apply geometric augmentation to images and adjust calibrations and trajectories.
        
        Args:
            images: Dict[camera_id -> Tensor[B, T, C, H, W]]
            calibrations: Dict[camera_id -> Tensor[B, calibration_params]]
            trajectories: Tensor[B, N, trajectory_params]
            
        Returns:
            Tuple of (augmented_images, adjusted_calibrations, adjusted_trajectories)
        """
        # Random scaling
        if self.config.random_scale[0] < self.config.random_scale[1]:
            scale = torch.rand(1) * (self.config.random_scale[1] - self.config.random_scale[0]) + self.config.random_scale[0]
            images = {k: F.resize(v, scale_factor=scale.item()) for k, v in images.items()}
            calibrations = {k: self._adjust_calibration(v, scale=scale) for k, v in calibrations.items()}
            trajectories = self._adjust_trajectories(trajectories, scale=scale)
        
        # Random rotation
        if self.config.random_rotate > 0:
            angle = (torch.rand(1) * 2 - 1) * self.config.random_rotate
            images = {k: F.rotate(v, angle.item()) for k, v in images.items()}
            calibrations = {k: self._adjust_calibration(v, angle=angle) for k, v in calibrations.items()}
            trajectories = self._adjust_trajectories(trajectories, angle=angle)
            
        return images, calibrations, trajectories
    
    def _adjust_calibration(self, 
                          calibration: torch.Tensor,
                          scale: Optional[torch.Tensor] = None,
                          angle: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Adjust calibration parameters for geometric transformations."""
        # Implementation depends on your calibration format
        return calibration
    
    def _adjust_trajectories(self,
                           trajectories: torch.Tensor,
                           scale: Optional[torch.Tensor] = None,
                           angle: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Adjust trajectory parameters for geometric transformations."""
        # Implementation depends on your trajectory format
        return trajectories 