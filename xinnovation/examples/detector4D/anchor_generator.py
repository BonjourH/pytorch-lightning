import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from xinnovation.src.core.registry import COMPONENTS

__all__ = ["Anchor3DGenerator"]

@COMPONENTS.register_module()
class Anchor3DGenerator(nn.Module):
    """Generate 3D anchors for object detection.
    
    This class generates 3D anchors in the ego vehicle's coordinate system.
    The anchors are defined by their center position (x, y, z) and dimensions (length, width, height).
    """
    
    def __init__(self, 
                 anchor_ranges: List[float],
                 anchor_sizes: List[List[float]],
                 rotations: List[float],
                 match_threshold: float = 0.5,
                 unmatch_threshold: float = 0.35,
                 **kwargs):
        """Initialize the anchor generator.
        
        Args:
            anchor_ranges: List of [x_min, y_min, z_min, x_max, y_max, z_max]
            anchor_sizes: List of [length, width, height] for each anchor type
            rotations: List of rotation angles in radians
            match_threshold: IoU threshold for positive anchors
            unmatch_threshold: IoU threshold for negative anchors
        """
        super().__init__()
        self.anchor_ranges = torch.tensor(anchor_ranges)
        self.anchor_sizes = torch.tensor(anchor_sizes)
        self.rotations = torch.tensor(rotations)
        self.match_threshold = match_threshold
        self.unmatch_threshold = unmatch_threshold
        
        # Generate anchor centers
        self.anchor_centers = self._generate_anchor_centers()
        
        # Generate anchor corners
        self.anchor_corners = self._generate_anchor_corners()
        
    def _generate_anchor_centers(self) -> torch.Tensor:
        """Generate anchor centers in the ego vehicle's coordinate system.
        
        Returns:
            torch.Tensor: Anchor centers with shape [N, 3] where N is the number of anchors
        """
        x_min, y_min, z_min, x_max, y_max, z_max = self.anchor_ranges
        
        # Generate grid points
        x = torch.arange(x_min, x_max + 0.5, 0.5)
        y = torch.arange(y_min, y_max + 0.5, 0.5)
        z = torch.arange(z_min, z_max + 0.5, 0.5)
        
        # Create meshgrid
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        
        # Stack coordinates
        centers = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
        
        return centers
    
    def _generate_anchor_corners(self) -> torch.Tensor:
        """Generate anchor corners for each anchor type and rotation.
        
        Returns:
            torch.Tensor: Anchor corners with shape [N, 8, 3] where N is the number of anchors
        """
        num_centers = len(self.anchor_centers)
        num_sizes = len(self.anchor_sizes)
        num_rotations = len(self.rotations)
        
        # Initialize corners tensor
        corners = torch.zeros((num_centers * num_sizes * num_rotations, 8, 3))
        
        # Generate corners for each combination of center, size, and rotation
        for i, center in enumerate(self.anchor_centers):
            for j, size in enumerate(self.anchor_sizes):
                for k, rotation in enumerate(self.rotations):
                    idx = i * num_sizes * num_rotations + j * num_rotations + k
                    corners[idx] = self._generate_single_anchor_corners(center, size, rotation)
        
        return corners
    
    def _generate_single_anchor_corners(self, 
                                      center: torch.Tensor,
                                      size: torch.Tensor,
                                      rotation: float) -> torch.Tensor:
        """Generate corners for a single anchor.
        
        Args:
            center: Center coordinates [x, y, z]
            size: Anchor dimensions [length, width, height]
            rotation: Rotation angle in radians
            
        Returns:
            torch.Tensor: 8 corners with shape [8, 3]
        """
        l, w, h = size
        
        # Generate corners in local coordinate system
        corners = torch.tensor([
            [-l/2, -w/2, -h/2],  # Front bottom left
            [l/2, -w/2, -h/2],   # Front bottom right
            [l/2, w/2, -h/2],    # Back bottom right
            [-l/2, w/2, -h/2],   # Back bottom left
            [-l/2, -w/2, h/2],   # Front top left
            [l/2, -w/2, h/2],    # Front top right
            [l/2, w/2, h/2],     # Back top right
            [-l/2, w/2, h/2],    # Back top left
        ])
        
        # Apply rotation around z-axis
        cos_theta = torch.cos(rotation)
        sin_theta = torch.sin(rotation)
        rotation_matrix = torch.tensor([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
        
        corners = corners @ rotation_matrix.T
        
        # Translate to center
        corners = corners + center
        
        return corners
    
    def forward(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            batch_size: Batch size for the output tensors
            
        Returns:
            Dict containing:
                - centers: Anchor centers [B, N, 3]
                - corners: Anchor corners [B, N, 8, 3]
                - sizes: Anchor sizes [B, N, 3]
                - rotations: Anchor rotations [B, N]
        """
        # Expand to batch size
        centers = self.anchor_centers.unsqueeze(0).expand(batch_size, -1, -1)
        corners = self.anchor_corners.unsqueeze(0).expand(batch_size, -1, -1, -1)
        sizes = self.anchor_sizes.unsqueeze(0).expand(batch_size, -1, -1)
        rotations = self.rotations.unsqueeze(0).expand(batch_size, -1)
        
        return {
            'centers': centers,
            'corners': corners,
            'sizes': sizes,
            'rotations': rotations
        }
    
    @classmethod
    def build(cls, cfg: Dict) -> 'Anchor3DGenerator':
        """Build an anchor generator from config.
        
        Args:
            cfg: Configuration dictionary
            
        Returns:
            Anchor3DGenerator instance
        """
        return cls(**cfg) 