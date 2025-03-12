import torch
import pytest
from typing import Dict, List
import numpy as np

from base import SourceCameraId, TrajParamIndex, CameraParamIndex, EgoStateIndex, CameraType

from utils.pose_transform import get_transform_from_object_to_camera, camera_to_pixel


@pytest.mark.parametrize(
    "x, y, z, expected_u, expected_v",
    [
        (0, 0, 10, 480, 270),
        (2, 0, 10, 737.822, 270),
        (0, 1, 5, 480, 527.822),
        (1, 1, 10, 610.514, 400.514),
    ],
)
def test_front_left_camera_to_pixel(x, y, z, expected_u, expected_v):
    """
    Test function for camera_to_pixel.
    Projects points to image and checks if the computed pixels match expected values.
    """
    B, C, N, T, P = 1, 1, 1, 1, 1
    cam_points = torch.zeros(B, C, N, T, P, 3)

    calib_params = torch.zeros(B, C, CameraParamIndex.END_OF_INDEX)
    calib_params[..., CameraParamIndex.FX] = 1321.48
    calib_params[..., CameraParamIndex.FY] = 1321.48
    calib_params[..., CameraParamIndex.CX] = 480
    calib_params[..., CameraParamIndex.CY] = 270
    calib_params[..., CameraParamIndex.K1] = -0.62441600000000008
    calib_params[..., CameraParamIndex.K2] = 0.30044600000000005
    calib_params[..., CameraParamIndex.IMAGE_WIDTH] = 960
    calib_params[..., CameraParamIndex.IMAGE_HEIGHT] = 540
    calib_params[..., CameraParamIndex.CAMERA_TYPE] = CameraType.GENERAL_DISTORT

    # 计算投影
    cam_points[..., 0] = x
    cam_points[..., 1] = y
    cam_points[..., 2] = z

    pixels, _ = camera_to_pixel(cam_points, calib_params, False)
    # 计算误差
    expected_pixels = torch.tensor([expected_u, expected_v])
    diff = torch.norm(pixels - expected_pixels, dim=-1)

    assert diff < 1e-2


@pytest.mark.parametrize(
    "x, y, z, expected_u, expected_v",
    [
        (0, 0, 10, 484.308, 268.883),
        (2, 0, 10, 580.3, 268.883),
        (0, 1, 5, 484.308, 363.581),
        (1, 1, 10, 532.655, 316.579),
    ],
)
def test_front_center_camera_to_pixel(x, y, z, expected_u, expected_v):
    """
    Test function for camera_to_pixel.
    Projects points to image and checks if the computed pixels match expected values.
    """
    B, C, N, T, P = 1, 1, 1, 1, 1
    cam_points = torch.zeros(B, C, N, T, P, 3)

    calib_params = torch.zeros(B, C, CameraParamIndex.END_OF_INDEX)
    
    calib_params[..., CameraParamIndex.FX] = 487.07297888651942
    calib_params[..., CameraParamIndex.FY] = 480.50703577470631
    calib_params[..., CameraParamIndex.CX] = 484.30814595650594
    calib_params[..., CameraParamIndex.CY] = 268.88344951522936
    calib_params[..., CameraParamIndex.K1] = -0.041464596584104653
    calib_params[..., CameraParamIndex.K2] = 0.0090031786097134383
    calib_params[..., CameraParamIndex.K3] = -0.013618604927618041
    calib_params[..., CameraParamIndex.K4] = 0.0037757395282444445
    calib_params[..., CameraParamIndex.IMAGE_WIDTH] = 960
    calib_params[..., CameraParamIndex.IMAGE_HEIGHT] = 540
    calib_params[..., CameraParamIndex.CAMERA_TYPE] = CameraType.FISHEYE

    # 计算投影
    cam_points[..., 0] = x
    cam_points[..., 1] = y
    cam_points[..., 2] = z

    pixels, _ = camera_to_pixel(cam_points, calib_params, False)
    # 计算误差
    expected_pixels = torch.tensor([expected_u, expected_v])
    diff = torch.norm(pixels - expected_pixels, dim=-1)

    assert diff < 1e-2

    
    
    
    
    
    
    