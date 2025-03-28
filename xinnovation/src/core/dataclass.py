import torch
from enum import IntEnum
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import numpy as np


# CAMERA RELATED 
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
    
class CameraType(IntEnum):
    UNKNOWN = 0
    PINHOLE = 1
    FISHEYE = 2
    GENERAL_DISTORT = 3
    OMNIDIRECTIONAL = 4
    
class CameraParamIndex(IntEnum):
    CAMERA_ID = 0
    CAMERA_TYPE = 1
    IMAGE_WIDTH = 2
    IMAGE_HEIGHT = 3
    FX = 4
    FY = 5
    CX = 6
    CY = 7
    K1 = 8
    K2 = 9
    K3 = 10
    K4 = 11
    P1 = 12
    P2 = 13
    R_EGO_TO_CAMERA_11 = 14
    R_EGO_TO_CAMERA_12 = 15
    R_EGO_TO_CAMERA_13 = 16
    R_EGO_TO_CAMERA_21 = 17
    R_EGO_TO_CAMERA_22 = 18
    R_EGO_TO_CAMERA_23 = 19
    R_EGO_TO_CAMERA_31 = 20
    R_EGO_TO_CAMERA_32 = 21
    R_EGO_TO_CAMERA_33 = 22
    T_EGO_TO_CAMERA_X = 23
    T_EGO_TO_CAMERA_Y = 24
    T_EGO_TO_CAMERA_Z = 25
    R_CAMERA_TO_EGO_11 = 26
    R_CAMERA_TO_EGO_12 = 27
    R_CAMERA_TO_EGO_13 = 28
    R_CAMERA_TO_EGO_21 = 29
    R_CAMERA_TO_EGO_22 = 30
    R_CAMERA_TO_EGO_23 = 31
    R_CAMERA_TO_EGO_31 = 32
    R_CAMERA_TO_EGO_32 = 33
    R_CAMERA_TO_EGO_33 = 34
    T_CAMERA_TO_EGO_X = 35
    T_CAMERA_TO_EGO_Y = 36
    T_CAMERA_TO_EGO_Z = 37
    END_OF_INDEX = 38

# EGO STATE RELATED
class EgoStateIndex(IntEnum):
    TIMESTAMP = 0
    X = 1
    Y = 2
    YAW = 3
    PITCH_CORRECTION = 4
    VX = 5
    VY = 6
    AX = 7
    AY = 8
    YAW_RATE = 9
    END_OF_INDEX = 10    
    
# OBSTACLE RELATED 
class ObjectType(IntEnum):
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
    END_OF_INDEX = 12
    
    def __str__(self):
        return self.name
    
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
    HAS_OBJECT = 11 # The below are object attributes
    STATIC = 12
    OCCLUDED = 13
    CAR = 14        # The below are object type
    SUV = 15
    LIGHTTRUCK = 16
    TRUCK = 17
    BUS = 18
    PEDESTRIAN = 19
    BICYCLE = 20
    MOTO = 21
    CYCLIST = 22
    MOTORCYCLIST = 23
    CONE = 24
    BACKGROUND = 25
    END_OF_INDEX = 26
    
    def __str__(self):
        return self.name
    
    def __int__(self):
        return self.value
    
    @classmethod
    def object_type_to_index(cls, obj_type: str) -> int:
        """Convert ObjectType to index."""
        return cls[obj_type].value


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
    def to_dict(self) -> Dict:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "vx": self.vx,
            "vy": self.vy,
            "vz": self.vz,
            "ax": self.ax,
            "ay": self.ay,
            "az": self.az
        }
    
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
    
    @property
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "motion": self.motion.to_dict,
            "yaw": self.yaw,
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "object_type": self.object_type.name,
            "static": self.static,
        }
    
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
    
def tensor_to_object_type(tensor: torch.Tensor) -> ObjectType:
    """Convert tensor to ObjectType."""
    index = int(torch.argmax(tensor[TrajParamIndex.CAR:])) + TrajParamIndex.CAR
    return ObjectType[TrajParamIndex(index).name]
    
def tensor_to_trajectory(traj_params: torch.Tensor, traj_id: int = 0, t0: float = 0.0) -> ObstacleTrajectory:
    """Convert trajectory parameters tensor to ObstacleTrajectory object.
    
    Args:
        traj_params: Tensor[TrajParamIndex.END_OF_INDEX] - Trajectory parameters
        traj_id: Optional trajectory ID
        t0: Optional reference timestamp
        
    Returns:
        ObstacleTrajectory object
    """
    # Create Point3DAccMotion object
    motion = Point3DAccMotion(
        x=float(traj_params[TrajParamIndex.X]),
        y=float(traj_params[TrajParamIndex.Y]),
        z=float(traj_params[TrajParamIndex.Z]),
        vx=float(traj_params[TrajParamIndex.VX]),
        vy=float(traj_params[TrajParamIndex.VY]),
        vz=0.0,  # Z velocity not predicted
        ax=float(traj_params[TrajParamIndex.AX]),
        ay=float(traj_params[TrajParamIndex.AY]),
        az=0.0   # Z acceleration not predicted
    )
    
    traj_params[TrajParamIndex.HAS_OBJECT:] = torch.sigmoid(traj_params[TrajParamIndex.HAS_OBJECT:])
    
    # Create ObstacleTrajectory object
    return ObstacleTrajectory(
        id=traj_id,
        motion=motion,
        yaw=float(traj_params[TrajParamIndex.YAW]),
        length=float(traj_params[TrajParamIndex.LENGTH]),
        width=float(traj_params[TrajParamIndex.WIDTH]),
        height=float(traj_params[TrajParamIndex.HEIGHT]),
        object_type=tensor_to_object_type(traj_params),
        t0=t0,
        static=bool(traj_params[TrajParamIndex.STATIC] > 0.5),
        valid=bool(traj_params[TrajParamIndex.HAS_OBJECT] > 0.5)
    )
    