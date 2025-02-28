import torch
from enum import IntEnum
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import numpy as np


class ObjectType(IntEnum):
    """Object type enumeration."""
    UNKNOWN = 0
    CAR = 1
    SUV = 2
    LIGHTTRUCK = 3
    TRUCK = 4
    BUS = 5
    PEDESTRIAN = 6
    BICYCLE = 7
    MOTO = 8
    CYCLIST = 9
    MOTORCYCLIST = 10
    CONE = 11
    BARRIER = 12

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    def __int__(self):
        return self.value


class SourceCameraId(IntEnum):
    NO_CAMERA = 0
    FRONT_LEFT_CAMERA = 1
    FRONT_RIGHT_CAMERA = 2
    FRONT_CENTER_CAMERA = 3
    SIDE_LEFT_CAMERA = 4
    SIDE_RIGHT_CAMERA = 5
    REAR_LEFT_CAMERA = 6
    REAR_RIGHT_CAMERA = 7
    FRONT_LEFT_TELE_CAMERA = 10
    FRONT_RIGHT_TELE_CAMERA = 11
    FRONT_CENTER_TELE_CAMERA = 12
    REAR_LEFT_TELE_CAMERA = 13
    REAR_RIGHT_TELE_CAMERA = 14
    STITCH_CAMERA = 20
    BEV_CAMERA = 21
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    def __int__(self):
        return self.value


class CameraType(IntEnum):
    UNKNOWN = 0
    PINHOLE = 1
    FISHEYE = 2
    GENERAL_DISTORT = 3
    OMNIDIRECTIONAL = 4
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    def __int__(self):
        return self.value
    

@dataclass
class CameraCalibration:
    camera_id: Optional[SourceCameraId] = None
    camera_type: Optional[CameraType] = None
    intrinsic: Optional[torch.Tensor] = None
    distortion: Optional[torch.Tensor] = None
    extrinsic: Optional[torch.Tensor] = None # from vehicle coordinate to camera coordinate
    
    
@dataclass
class Object3D:
    """3D object annotation."""
    id: str
    type: ObjectType
    position: np.ndarray  # [x, y, z]
    dimensions: np.ndarray  # [length, width, height]
    yaw: float # in radians
    velocity: Optional[np.ndarray] = None  # [vx, vy]
    valid: bool = True

    @property
    def corners(self) -> np.ndarray:
        """Get 8 corners of 3D bounding box."""
        l, w, h = self.dimensions
        corners = np.array([
            [-l/2, -w/2, -h/2],
            [l/2, -w/2, -h/2],
            [l/2, w/2, -h/2],
            [-l/2, w/2, -h/2],
            [-l/2, -w/2, h/2],
            [l/2, -w/2, h/2],
            [l/2, w/2, h/2],
            [-l/2, w/2, h/2]
        ])
        
        rot = np.array([
            [np.cos(self.yaw), -np.sin(self.yaw), 0],
            [np.sin(self.yaw), np.cos(self.yaw), 0],
            [0, 0, 1]
        ])
        corners = corners @ rot.T
        corners += self.position
        return corners
    
    
    