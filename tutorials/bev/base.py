from enum import IntEnum
from typing import Dict, List


class CameraType(IntEnum):
    PINHOLE = 0
    GENERAL_DISTORT = 1
    FISHEYE = 2
    OMNIDIRECTIONAL = 3


class SourceCameraId(IntEnum):
    FRONT_LEFT_CAMERA = 0
    FRONT_CENTER_CAMERA = 1
    FRONT_RIGHT_CAMERA = 2
    SIDE_LEFT_CAMERA = 3
    SIDE_RIGHT_CAMERA = 4
    REAR_LEFT_CAMERA = 5
    REAR_RIGHT_CAMERA = 6


def is_front_camera(camera_id: SourceCameraId) -> bool:
    return camera_id in [SourceCameraId.FRONT_LEFT_CAMERA, SourceCameraId.FRONT_CENTER_CAMERA, SourceCameraId.FRONT_RIGHT_CAMERA]


def is_side_camera(camera_id: SourceCameraId) -> bool:
    return camera_id in [SourceCameraId.SIDE_LEFT_CAMERA, SourceCameraId.SIDE_RIGHT_CAMERA]


def is_rear_camera(camera_id: SourceCameraId) -> bool:
    return camera_id in [SourceCameraId.REAR_LEFT_CAMERA, SourceCameraId.REAR_RIGHT_CAMERA]
    
    
class SpeedStatus(IntEnum):
    NOT_CONVERGE = 0
    NOISY = 1
    ROBUST = 2
    
    
class ObjectType(IntEnum):
    CAR = 0
    PEDESTRIAN = 1
    BICYCLE = 2
    TRUCK = 3
    BUS = 4
    MOTO = 5
    BARRIER = 6
    CONE = 7
    UNKNOWN = 8
    HEAVY_EQUIPMENT = 9
    MOVABLE_SIGN = 10
    EMERGENCY_VEHICLE = 11
    LICENSE_PLATE = 12
    SUV = 13
    LIGHTTRUCK = 14


class PinholeCameraParamIndex(IntEnum):
    TYPE = 0
    QX = 1
    QY = 2
    QZ = 3
    QW = 4
    TX = 5
    TY = 6
    TZ = 7
    WIDTH = 8
    HEIGHT = 9
    FX = 10
    FY = 11
    CX = 12
    CY = 13
    

class DistortCameraParamIndex(IntEnum):
    TYPE = 0
    QX = 1
    QY = 2
    QZ = 3
    QW = 4
    TX = 5
    TY = 6
    TZ = 7
    WIDTH = 8
    HEIGHT = 9
    FX = 10
    FY = 11
    CX = 12
    CY = 13
    K1 = 14
    K2 = 15
    P1 = 16
    P2 = 17
    K3 = 18


class FisheyeCameraParamIndex(IntEnum):
    TYPE = 0
    QX = 1
    QY = 2
    QZ = 3
    QW = 4
    TX = 5
    TY = 6
    TZ = 7
    WIDTH = 8
    HEIGHT = 9
    FX = 10
    FY = 11
    CX = 12
    CY = 13
    K1 = 14
    K2 = 15
    K3 = 16
    K4 = 17
    
    
class EgoStateIndex(IntEnum):
    TIMESTAMP = 0
    X = 1
    Y = 2
    Z = 3
    QX = 4
    QY = 5
    QZ = 6
    QW = 7
    SPEED = 8
    YAW_RATE = 9
    ACCELERATION = 10
    PITCH = 11
    END_INDEX = 12
    
    
class ObstacleStateIndex(IntEnum):
    ID = 0
    TYPE = 1
    X = 2
    Y = 3
    Z = 4
    QX = 5
    QY = 6
    QZ = 7
    QW = 8
    LENGTH = 9
    WIDTH = 10
    HEIGHT = 11
    VX = 12
    VY = 13
    VZ = 14
    SPEED_STATUS = 15
    END_INDEX = 16


def convert_calib_dict_to_vec(calib: Dict) -> List:
    List = [0 for _ in range(20)]
    List[0] = CameraType[calib['type']]
    List[1] = calib['extrinsic']['qx']
    List[2] = calib['extrinsic']['qy']
    List[3] = calib['extrinsic']['qz']
    List[4] = calib['extrinsic']['qw']
    List[5] = calib['extrinsic']['tx']
    List[6] = calib['extrinsic']['ty']
    List[7] = calib['extrinsic']['tz']
    List[8] = calib['intrinsic']['width']
    List[9] = calib['intrinsic']['height']
    List[10] = calib['intrinsic']['fx']
    List[11] = calib['intrinsic']['fy']
    List[12] = calib['intrinsic']['cx']
    List[13] = calib['intrinsic']['cy']
    if calib['type'] == 'GENERAL_DISTORT':
        List[14] = calib['intrinsic']['k1']
        List[15] = calib['intrinsic']['k2']
        List[16] = calib['intrinsic']['p1']
        List[17] = calib['intrinsic']['p2']
        List[18] = calib['intrinsic']['k3']
    elif calib['type'] == 'FISHEYE':
        List[14] = calib['intrinsic']['k1']
        List[15] = calib['intrinsic']['k2']
        List[16] = calib['intrinsic']['k3']
        List[17] = calib['intrinsic']['k4']
    return List


def convert_ego_state_dict_to_vec(ego_state: Dict) -> List:
    List = [0 for _ in range(EgoStateIndex.END_INDEX)]
    List[EgoStateIndex.TIMESTAMP] = ego_state['timestamp']
    List[EgoStateIndex.X] = ego_state['x']
    List[EgoStateIndex.Y] = ego_state['y']
    List[EgoStateIndex.Z] = ego_state['z']
    List[EgoStateIndex.QX] = ego_state['qx']
    List[EgoStateIndex.QY] = ego_state['qy']
    List[EgoStateIndex.QZ] = ego_state['qz']
    List[EgoStateIndex.QW] = ego_state['qw']
    List[EgoStateIndex.SPEED] = ego_state['speed']
    List[EgoStateIndex.YAW_RATE] = 0 # currently not used
    List[EgoStateIndex.ACCELERATION] = ego_state['acceleration']
    List[EgoStateIndex.PITCH] = ego_state['pitch']
    return List


def convert_obstable_3D_state_to_vec(obstacle_state: Dict) -> List:
    List = [0 for _ in range(ObstacleStateIndex.END_INDEX)]
    List[ObstacleStateIndex.ID] = obstacle_state['id']
    List[ObstacleStateIndex.TYPE] = ObjectType[obstacle_state['type']]
    List[ObstacleStateIndex.X] = obstacle_state['x']
    List[ObstacleStateIndex.Y] = obstacle_state['y']
    List[ObstacleStateIndex.Z] = obstacle_state['z']
    List[ObstacleStateIndex.QX] = obstacle_state['qx']
    List[ObstacleStateIndex.QY] = obstacle_state['qy']
    List[ObstacleStateIndex.QZ] = obstacle_state['qz']
    List[ObstacleStateIndex.QW] = obstacle_state['qw']
    List[ObstacleStateIndex.LENGTH] = obstacle_state['length']
    List[ObstacleStateIndex.WIDTH] = obstacle_state['width']
    List[ObstacleStateIndex.HEIGHT] = obstacle_state['height']
    List[ObstacleStateIndex.VX] = obstacle_state['vx']
    List[ObstacleStateIndex.VY] = obstacle_state['vy']
    List[ObstacleStateIndex.VZ] = obstacle_state['vz']
    List[ObstacleStateIndex.SPEED_STATUS] = SpeedStatus[obstacle_state['speed_status']]
    return List