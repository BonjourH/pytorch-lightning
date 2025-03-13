import torch
import math
import numpy as np

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
