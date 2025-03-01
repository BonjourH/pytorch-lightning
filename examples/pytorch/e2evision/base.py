import torch
from enum import IntEnum
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import numpy as np


class AttributeType(IntEnum):
    """Attribute type enumeration."""
    HAS_OBJECT = 0
    STATIC = 1
    OCCLUDED = 2
    END_OF_INDEX = 3


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
    

class TrajParamIndex(IntEnum):
    """Motion parameters index enumeration."""
    X = 0           # position x
    Y = 1           # position y
    Z = 2           # position z
    VX = 3          # velocity x
    VY = 4          # velocity y
    AX = 5          # acceleration x
    AY = 6          # acceleration y
    YAW = 7         # orientation
    LENGTH = 8      # dimensions
    WIDTH = 9
    HEIGHT = 10
    END_OF_INDEX = 11

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    def __int__(self):
        return self.value


@dataclass
class Point3DAccMotion:
    # position
    x: float
    y: float
    z: float
    # velocity
    vx: float
    vy: float
    vz: float
    # acceleration
    ax: float
    ay: float
    az: float
    
    def position_at(self, t: float) -> np.ndarray:
        """Get position at time t using constant acceleration equation."""
        # x = x0 + v0*t + (1/2)*a*t^2
        x = self.x + self.vx * t + 0.5 * self.ax * t * t
        y = self.y + self.vy * t + 0.5 * self.ay * t * t
        z = self.z + self.vz * t + 0.5 * self.az * t * t
        return np.array([x, y, z])
    
    def velocity_at(self, t: float) -> np.ndarray:
        """Get velocity at time t."""
        # v = v0 + a*t
        vx = self.vx + self.ax * t
        vy = self.vy + self.ay * t
        vz = self.vz + self.az * t
        return np.array([vx, vy, vz])
    
    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz])
    
    @property
    def acceleration(self) -> np.ndarray:
        return np.array([self.ax, self.ay, self.az])


@dataclass
class ObstacleTrajectory:
    """Trajectory class for tracking obstacles with motion prediction."""
    id: int
    motion: Point3DAccMotion
    yaw: float
    length: float
    width: float
    height: float
    object_type: ObjectType
    t0: float = 0         # reference timestamp
    static: bool = False
    valid: bool = True
    
    @property
    def position(self) -> np.ndarray:
        return self.motion.position
    
    @property
    def velocity(self) -> np.ndarray:
        return self.motion.velocity
    
    @property
    def acceleration(self) -> np.ndarray:
        return self.motion.acceleration
    
    @property
    def dimensions(self) -> np.ndarray:
        return np.array([self.length, self.width, self.height])
    
    def center(self, timestamp: float) -> np.ndarray:
        """Get center position at a given timestamp."""
        t = timestamp - self.t0
        return self.motion.position_at(t)
    
    def velocity_at(self, timestamp: float) -> np.ndarray:
        """Get velocity at a given timestamp."""
        t = timestamp - self.t0
        return self.motion.velocity_at(t)
    
    def yaw_at(self, timestamp: float) -> float:
        """Get yaw angle at a given timestamp.
        When speed > 0.2, use velocity direction as yaw.
        Otherwise use initial yaw."""
        v = self.velocity_at(timestamp)
        speed = np.linalg.norm(v[:2])  # Only consider x-y plane for yaw
        if speed > 0.2:
            return np.arctan2(v[1], v[0])
        return self.yaw
    
    def corners(self, timestamp: Optional[float] = None) -> np.ndarray:
        """Get 8 corners of 3D bounding box at a given timestamp.
        If timestamp is None, use current position."""
        if timestamp is not None:
            center = self.center(timestamp)
            yaw = self.yaw_at(timestamp)
        else:
            center = self.position
            yaw = self.yaw
            
        l, w, h = self.dimensions
        
        # Create corner offsets from center
        corner_offsets = np.array([
            [-l/2, -w/2, -h/2],  # 前左下
            [l/2, -w/2, -h/2],   # 前右下
            [l/2, w/2, -h/2],    # 后右下
            [-l/2, w/2, -h/2],   # 后左下
            [-l/2, -w/2, h/2],   # 前左上
            [l/2, -w/2, h/2],    # 前右上
            [l/2, w/2, h/2],     # 后右上
            [-l/2, w/2, h/2]     # 后左上
        ])
        
        # Apply yaw rotation
        rot = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        corners = corner_offsets @ rot.T + center
        return corners
    