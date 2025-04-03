import torch
import math
import numpy as np
from xinnovation.src.core import TrajParamIndex

def quaternion2RotationMatix(qw, qx, qy, qz) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix
    :param qw: w part of quaternion
    :param qx: x part of quaternion
    :param qy: y part of quaternion
    :param qz: z part of quaternion
    :return: rotation matrix
    """
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw = qw / norm
    qx = qx / norm
    qy = qy / norm
    qz = qz / norm
    
    return torch.tensor([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])

def generate_unit_cube_points(num_points: int = 25):
    """Generate sample points on faces of unit cube.
    
    Args:
        num_points: Number of points to sample on each face
        
    Returns:
        Tensor of shape [3, num_points*6 + 1] containing sampled points
    """
    points = []
    points_per_face = int(np.sqrt(num_points))  # e.g., 5 for 25 points per face
    
    # Sample points on each face
    for dim in range(3):  # x, y, z
        for sign in [-1, 1]:  # negative and positive faces
            # Create grid on face
            if dim == 0:  # yz plane
                y = torch.linspace(-1, 1, points_per_face)
                z = torch.linspace(-1, 1, points_per_face)
                grid_y, grid_z = torch.meshgrid(y, z, indexing='ij')
                x = torch.full_like(grid_y, sign)
                points.append(torch.stack([x, grid_y, grid_z], dim=-1))
                
            elif dim == 1:  # xz plane
                x = torch.linspace(-1, 1, points_per_face)
                z = torch.linspace(-1, 1, points_per_face)
                grid_x, grid_z = torch.meshgrid(x, z, indexing='ij')
                y = torch.full_like(grid_x, sign)
                points.append(torch.stack([grid_x, y, grid_z], dim=-1))
                
            else:  # xy plane
                x = torch.linspace(-1, 1, points_per_face)
                y = torch.linspace(-1, 1, points_per_face)
                grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
                z = torch.full_like(grid_x, sign)
                points.append(torch.stack([grid_x, grid_y, z], dim=-1))
    
    points = torch.cat([p.reshape(-1, 3) for p in points], dim=0)
    # add origin point
    points = torch.cat([torch.zeros(1, 3), points], dim=0)
    points = points.transpose(0, 1) # [3, P]
    points /= 2.0
    return points


def generate_bbox_corners_points(with_origin: bool=True) -> torch.Tensor:
    """Generate sample points on corners of 3D bounding box.
    
    Returns:
        Tensor of shape [3, P] containing sampled points
    """
    corners = torch.tensor([
        [0.5, 0.5, -0.5], # front left bottom
        [0.5, -0.5, -0.5], # front right bottom
        [-0.5, -0.5, -0.5], # rear right bottom
        [-0.5, 0.5, -0.5], # rear left bottom
        [0.5, 0.5, 0.5], # front left top
        [0.5, -0.5, 0.5], # front right top
        [-0.5, -0.5, 0.5], # rear right top
        [-0.5, 0.5, 0.5], # rear left top
    ]).transpose(0, 1) # [3, 8]
    if with_origin:
        corners = torch.cat([torch.zeros(3, 1), corners], dim=1)
    return corners

def sample_bbox_edge_points(n_points_per_edge: int, include_corners: bool=True) -> torch.Tensor:
    """
    Sample points uniformly along each edge of a 3D bounding box.
    
    Args:
        n_points_per_edge: Number of points to sample on each edge (excluding corners)
        include_corners: Whether to include corner points in the output
        
    Returns:
        Tensor containing sampled points on edges of the bounding box
        shape: [3, P]
    """
    # Define the 8 corners of the bounding box
    corners = torch.tensor([
        [0.5, 0.5, -0.5],   # 0: front left bottom
        [0.5, -0.5, -0.5],  # 1: front right bottom
        [-0.5, -0.5, -0.5], # 2: rear right bottom
        [-0.5, 0.5, -0.5],  # 3: rear left bottom
        [0.5, 0.5, 0.5],    # 4: front left top
        [0.5, -0.5, 0.5],   # 5: front right top
        [-0.5, -0.5, 0.5],  # 6: rear right top
        [-0.5, 0.5, 0.5],   # 7: rear left top
    ])
    
    # Define the 12 edges by pairs of corner indices
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # connecting edges
    ]
    
    # List to collect all points
    all_points = []
    
    # Handle corner points if requested
    if include_corners:
        all_points.append(corners)
    
    # Generate points along each edge
    for i, j in edges:
        # Get the start and end points of this edge
        start, end = corners[i], corners[j]
        
        # Create evenly spaced points along the edge (excluding endpoints)
        if n_points_per_edge > 0:
            t = torch.linspace(0, 1, n_points_per_edge + 2)[1:-1]
            
            # Linear interpolation
            points = start + t.unsqueeze(1) * (end - start)
            all_points.append(points)
    
    points = torch.cat(all_points, dim=0).transpose(0, 1)
    
    # Combine all points into a single tensor (3, P)
    return points

def get_motion_param_range()->torch.Tensor:
    """Get parameter ranges for normalization.
    Returns:
        Tensor of shape [TrajParamIndex.HEIGHT + 1, 2] containing min/max values
    """
    param_range = torch.zeros(TrajParamIndex.HEIGHT + 1, 2)
    
    # Position ranges (in meters)
    param_range[TrajParamIndex.X] = torch.tensor([-80.0, 250.0])
    param_range[TrajParamIndex.Y] = torch.tensor([-10.0, 10.0])
    param_range[TrajParamIndex.Z] = torch.tensor([-3.0, 5.0])
    
    # Velocity ranges (in m/s)
    param_range[TrajParamIndex.VX] = torch.tensor([-40.0, 40.0])
    param_range[TrajParamIndex.VY] = torch.tensor([-5.0, 5.0])
    
    # Acceleration ranges (in m/s^2)
    param_range[TrajParamIndex.AX] = torch.tensor([-5.0, 5.0])
    param_range[TrajParamIndex.AY] = torch.tensor([-2.0, 2.0])
    
    # Yaw range (in radians)
    param_range[TrajParamIndex.YAW] = torch.tensor([-np.pi, np.pi])
    
    # Dimension ranges (in meters)
    param_range[TrajParamIndex.LENGTH] = torch.tensor([0.2, 25.0])
    param_range[TrajParamIndex.WIDTH] = torch.tensor([0.2, 3.0])
    param_range[TrajParamIndex.HEIGHT] = torch.tensor([0.5, 5.0])
    
    return param_range