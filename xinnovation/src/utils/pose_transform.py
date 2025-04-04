import torch
from xinnovation.src.core import TrajParamIndex, EgoStateIndex, CameraParamIndex, CameraType
from xinnovation.src.utils.latency_utils import measure_gpu_latency, measure_average_gpu_latency
from typing import Tuple
from xinnovation.src.utils.debug_utils import check_nan_or_inf
import numpy as np

check_abnormal = False

def get_transform_from_object_to_camera(
    trajs: torch.Tensor,
    calibrations: torch.Tensor,
    ego_states: torch.Tensor
) -> torch.Tensor:
    """
    Get the transformation from sequential local object coordinates to sequential camera coordinates.
    Args:
        trajs: torch.Tensor[B, N, TrajParamIndex.END_OF_INDEX]
        calibrations: torch.Tensor[B, C, CameraParamIndex.END_OF_INDEX]
        ego_states: torch.Tensor[B, T, EgoStateIndex.END_OF_INDEX]
    Returns:
        # the transform from local object coordinates to camera frame
        # R_ob_to_camera: torch.Tensor[B, C, N, T, 3, 3]
        # t_ob_to_camera: torch.Tensor[B, C, N, T, 3, 1]
    """
    B, N, _ = trajs.shape
    C = calibrations.shape[1]
    T = ego_states.shape[1]
    
    # ob_i => ego_temporal => ego_i => camera
    # 1. get transformation T_ob_to_ego_temporal from sequential local object coordinates to temporal ego coordinates
    # the motion equation of object is built in temporal ego coordinates
    dt = ego_states[:, :, EgoStateIndex.TIMESTAMP] - ego_states[:, T-1, EgoStateIndex.TIMESTAMP].unsqueeze(1) # [B, T]
    dt = dt.unsqueeze(1) # [B, 1, T]
    
    pos_x = trajs[..., TrajParamIndex.X].unsqueeze(-1)  # [B, N, 1]
    pos_y = trajs[..., TrajParamIndex.Y].unsqueeze(-1)  # [B, N, 1]
    pos_z = trajs[..., TrajParamIndex.Z].unsqueeze(-1)  # [B, N, 1]
    
    vel_x = trajs[..., TrajParamIndex.VX].unsqueeze(-1)  # [B, N, 1]
    vel_y = trajs[..., TrajParamIndex.VY].unsqueeze(-1)  # [B, N, 1]
    
    acc_x = trajs[..., TrajParamIndex.AX].unsqueeze(-1)  # [B, N, 1]
    acc_y = trajs[..., TrajParamIndex.AY].unsqueeze(-1)  # [B, N, 1]
    
    traj_cos_yaw = trajs[..., TrajParamIndex.COS_YAW].unsqueeze(-1)  # [B, N, 1]
    traj_sin_yaw = trajs[..., TrajParamIndex.SIN_YAW].unsqueeze(-1)  # [B, N, 1]
    pred_yaw = torch.atan2(traj_sin_yaw, traj_cos_yaw)  # [B, N, 1]
    
    # Calculate position at time t using motion model (constant acceleration)
    # x(t) = x0 + v0*t + 0.5*a*t^2
    pos_x_t = pos_x + vel_x * dt + 0.5 * acc_x * dt * dt # [B, N, T]
    pos_y_t = pos_y + vel_y * dt + 0.5 * acc_y * dt * dt # [B, N, T]
    pos_z_t = pos_z  # Assume constant height
    
    # Calculate velocity at time t
    # v(t) = v0 + a*t
    vel_x_t = vel_x + acc_x * dt # [B, N, T]
    vel_y_t = vel_y + acc_y * dt # [B, N, T]
    
    # Determine yaw based on velocity or use initial yaw
    speed_t = torch.sqrt(vel_x_t*vel_x_t + vel_y_t*vel_y_t)
    velocity_yaw = torch.atan2(vel_y_t, vel_x_t)
    
    # If speed is sufficient, use velocity direction; otherwise use provided yaw
    yaw_t = torch.where(speed_t > 0.2, velocity_yaw, pred_yaw)
    
    # now we have pos_x_t, pos_y_t, pos_z_t, yaw_t, which is the Transform from object to current ego frame
    # shape: [B, N, T]
    
    # 2. get transformation T_ego_temporal_to_ego_previous from current ego frame to previous ego frame
    # ego states is the transform from current ego frame to previous ego frame
    ego_yaw = ego_states[..., EgoStateIndex.YAW].unsqueeze(1)  # [B, 1, T]
    ego_x = ego_states[..., EgoStateIndex.X].unsqueeze(1)  # [B, 1, T]
    ego_y = ego_states[..., EgoStateIndex.Y].unsqueeze(1)  # [B, 1, T]
    
    # 3. get transformation T_ob_to_ego_previous from local object coordinates to previous ego frame
    cos_yaw = torch.cos(ego_yaw)
    sin_yaw = torch.sin(ego_yaw)
    tx_to_previous_ego = pos_x_t * cos_yaw - pos_y_t * sin_yaw + ego_x # [B, N, T]
    ty_to_previous_ego = pos_x_t * sin_yaw + pos_y_t * cos_yaw + ego_y # [B, N, T]
    
    yaw_to_previous_ego = yaw_t + ego_yaw # [B, N, T]

    cos1, sin1 = torch.cos(yaw_to_previous_ego), torch.sin(yaw_to_previous_ego)
      
    zeros = torch.zeros_like(yaw_to_previous_ego)
    ones = torch.ones_like(yaw_to_previous_ego)
    
    R_ob_to_ego_previous = torch.stack([
        cos1, -sin1, zeros,
        sin1, cos1, zeros,
        zeros, zeros, ones
    ], dim=-1).reshape(B, N, T, 3, 3)
    
    t_ob_to_ego_previous = torch.stack([
        tx_to_previous_ego,
        ty_to_previous_ego,
        pos_z_t * torch.ones_like(tx_to_previous_ego)
    ], dim=-1).reshape(B, N, T, 3, 1)
    
    # 4. calculate the dynamic camera extrinsic parameters with pitch correction
    pitch_correction = -ego_states[..., EgoStateIndex.PITCH_CORRECTION] # [B, T]
    cos_pitch, sin_pitch = torch.cos(pitch_correction), torch.sin(pitch_correction)
    ones, zeros = torch.ones_like(cos_pitch), torch.zeros_like(cos_pitch)
    R_perturbation = torch.stack([
        cos_pitch, zeros, sin_pitch,
        zeros, ones, zeros,
        -sin_pitch, zeros, cos_pitch
    ], dim=-1).reshape(B, T, 3, 3).unsqueeze(1)
    
    R_ego_to_camera = calibrations[..., CameraParamIndex.R_EGO_TO_CAMERA_11:
                                   CameraParamIndex.R_EGO_TO_CAMERA_33+1].reshape(B, C, 3, 3).unsqueeze(2)
    t_cam_to_ego = calibrations[..., CameraParamIndex.T_CAMERA_TO_EGO_X:
                                CameraParamIndex.T_CAMERA_TO_EGO_Z+1].reshape(B, C, 3, 1).unsqueeze(2)
    
    R_ego_to_cam_refined = R_ego_to_camera @ R_perturbation # [B, C, T, 3, 3]
    t_ego_to_cam_refined = -R_ego_to_cam_refined @ t_cam_to_ego # [B, C, T, 3, 1]
    
    # 5. get the final transform from local object coordinates to camera frame
    R_ob_to_ego_previous = R_ob_to_ego_previous.unsqueeze(1) # [B, 1, N, T, 3, 3]
    t_ob_to_ego_previous = t_ob_to_ego_previous.unsqueeze(1) # [B, 1, N, T, 3, 1]
    R_ego_to_cam_refined = R_ego_to_cam_refined.unsqueeze(2) # [B, C, 1, T, 3, 3]
    t_ego_to_cam_refined = t_ego_to_cam_refined.unsqueeze(2) # [B, C, 1, T, 3, 1]
    R_ob_to_camera = R_ego_to_cam_refined @ R_ob_to_ego_previous # [B, C, N, T, 3, 3]
    t_ob_to_camera = R_ego_to_cam_refined @ t_ob_to_ego_previous + t_ego_to_cam_refined # [B, C, N, T, 3, 1]
    
    tr_ob_to_camera = torch.cat([R_ob_to_camera, t_ob_to_camera], dim=-1) # [B, C, N, T, 3, 4]
    
    return tr_ob_to_camera


def camera_to_pixel(cam_points: torch.Tensor, calib_params: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Project points to image.
    Args:
        cam_points: torch.Tensor[B, C, N, T, P, 3]
        calib_params: torch.Tensor[B, C, CameraParamIndex.END_OF_INDEX]
    Returns:
        pixels: torch.Tensor[B*T, N, C, P, 2]
    """
    B, C, N, T, P, _ = cam_points.shape
    cam_points = cam_points.view(B, C, -1, 3)
    
    # Extract camera parameters
    camera_type = calib_params[..., CameraParamIndex.CAMERA_TYPE].long() # [B, C]
    
    # Get intrinsic parameters
    fx = calib_params[..., CameraParamIndex.FX].unsqueeze(2)  # [B, C, ]
    fy = calib_params[..., CameraParamIndex.FY].unsqueeze(2)  # [B, C, 1]
    cx = calib_params[..., CameraParamIndex.CX].unsqueeze(2)  # [B, C, 1]
    cy = calib_params[..., CameraParamIndex.CY].unsqueeze(2)  # [B, C, 1]
    
    # Distortion parameters
    k1 = calib_params[..., CameraParamIndex.K1].unsqueeze(2)  # [B, C, 1]
    k2 = calib_params[..., CameraParamIndex.K2].unsqueeze(2)  # [B, C, 1]
    k3 = calib_params[..., CameraParamIndex.K3].unsqueeze(2)  # [B, C, 1]
    k4 = calib_params[..., CameraParamIndex.K4].unsqueeze(2)  # [B, C, 1]
    p1 = calib_params[..., CameraParamIndex.P1].unsqueeze(2)  # [B, C, 1]
    p2 = calib_params[..., CameraParamIndex.P2].unsqueeze(2)  # [B, C, 1]

    # Check if points are behind the camera
    x_cam = cam_points[..., 0] # [B, C, NTP]
    y_cam = cam_points[..., 1] # [B, C, NTP]
    z_cam = cam_points[..., 2] # [B, C, NTP]
    
    
    # 1. calculate fisheye distorted coordinates
    r_square = x_cam * x_cam + y_cam * y_cam
    r = torch.sqrt(r_square)
    theta = torch.atan2(r, z_cam)
    r = torch.where(r == 0, torch.ones_like(r) * 1e-10, r)
    cos_alpha = x_cam / r
    sin_alpha = y_cam / r
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    
    distorted_theta = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
    x_distorted_fisheye = distorted_theta * cos_alpha
    y_distorted_fisheye = distorted_theta * sin_alpha
    
    # calculate the derivation of the distorted radius to filter the folded pixels
    # distorted_theta = theta * (1 + k1 * theta^2 + k2 * theta^4 + k3 * theta^6 + k4 * theta^8)
    # d(distorted_theta) / d(theta) = 1 + 3 * k1 * theta^2 + 5 * k2 * theta^4 + 7 * k3 * theta^6 + 9 * k4 * theta^8
    fisheye_derivation = 1 + 3 * k1 * theta2 + 5 * k2 * theta4 + 7 * k3 * theta6 + 9 * k4 * theta8
    folded_fisheye_pixel_mask = (fisheye_derivation < 0)
    

    x_distorted_fisheye[folded_fisheye_pixel_mask] = -1.0 # -1 is the invalid pixel
    y_distorted_fisheye[folded_fisheye_pixel_mask] = -1.0 # -1 is the invalid pixel
    
    #2.  calculate general distorted coordinates
    # Normalize coordinates
    z_cam = torch.where(z_cam == 0, torch.ones_like(z_cam) * 1e-5, z_cam)
    x_normalized = x_cam / z_cam 
    y_normalized = y_cam / z_cam
    normalized_r2 = x_normalized * x_normalized + y_normalized * y_normalized
    invalid_fov_mask = (normalized_r2 > 100) | (z_cam < 0)
    # 100 ~= tan(89.5), in general this type camera has no large FOV, here we mask the large
    # normalized_r2 to avoid the overflow in later calculation
    normalized_r2[invalid_fov_mask] = 0
    normalized_r4 = normalized_r2 * normalized_r2
    normalized_r6 = normalized_r4 * normalized_r2
    
    radial = 1 + k1 * normalized_r2 + k2 * normalized_r4 + k3 * normalized_r6
    
    # the distorted radius is r * (1 + k1 * r^2 + k2 * r^4 + k3 * r^6)
    # the derivation of the distorted radius is 1 + 3 * k1 * r^2 + 5 * k2 * r^4 + 7 * k3 * r^6
    distorted_r_derivation = 1 + 3 * k1 * normalized_r2 + 5 * k2 * normalized_r4 + 7 * k3 * normalized_r6
    folded_general_pixel_mask = (distorted_r_derivation < 0)
    
    dx = 2 * p1 * x_normalized * y_normalized + p2 * (normalized_r2 + 2 * x_normalized * x_normalized)
    dy = p1 * (normalized_r2 + 2 * y_normalized * y_normalized) + 2 * p2 * x_normalized * y_normalized
    
    x_distorted_general = x_normalized * radial + dx
    y_distorted_general = y_normalized * radial + dy

    general_invalid_mask = invalid_fov_mask | folded_general_pixel_mask
    x_distorted_general[general_invalid_mask] = -1.0 # -1 is the invalid pixel
    y_distorted_general[general_invalid_mask] = -1.0 # -1 is the invalid pixel
    check_nan_or_inf(x_distorted_general, active=check_abnormal, name="x_distorted_general")
    check_nan_or_inf(y_distorted_general, active=check_abnormal, name="y_distorted_general")
    
    # calculate pinhole distorted coordinates
    # no need to calculate
    # x_distorted_pinhole = x_normalized
    # y_distorted_pinhole = y_normalized
    
    # combine all distorted coordinates
    fish_eye_mask = (camera_type == CameraType.FISHEYE).unsqueeze(2)
    # general_distort_mask = (camera_type == CameraType.GENERAL_DISTORT).unsqueeze(2)
    # pinhole_mask = (camera_type == CameraType.PINHOLE).unsqueeze(2)
    
    x_distorted = torch.where(fish_eye_mask, x_distorted_fisheye, x_distorted_general)
    y_distorted = torch.where(fish_eye_mask, y_distorted_fisheye, y_distorted_general)
    
    # apply camera matrix
    x_pixel = fx * x_distorted + cx # [B, C, NTP]
    y_pixel = fy * y_distorted + cy
    check_nan_or_inf(x_pixel, active=check_abnormal, name="x_pixel")
    check_nan_or_inf(y_pixel, active=check_abnormal, name="y_pixel")

    # Get image dimensions
    img_width = calib_params[..., CameraParamIndex.IMAGE_WIDTH].unsqueeze(2)  # [B, C, 1]
    img_height = calib_params[..., CameraParamIndex.IMAGE_HEIGHT].unsqueeze(2)  # [B, C, 1]

    # get the invalid mask
    invalid_mask = (x_pixel < 0) | (x_pixel > img_width - 1) | (y_pixel < 0) | (y_pixel > img_height - 1) #| (z_cam <= 0)
    
    # normalize to [0, 1] for consistency with the visibility check in gather_point_features
    if normalize:
        x_norm = x_pixel / img_width
        y_norm = y_pixel / img_height
    else:
        x_norm = x_pixel
        y_norm = y_pixel

    check_nan_or_inf(x_norm, active=check_abnormal, name="x_norm")
    check_nan_or_inf(y_norm, active=check_abnormal, name="y_norm")
    
    points_2d = torch.stack([x_norm, y_norm], dim=-1) # [B, C, NTP, 2]
    points_2d[invalid_mask] = -1.0 # [B, C, NTP]
    points_2d = points_2d.view(B, C, N, T, P, 2)
    
    # readjust the dimension order: as we'll do feature sampling from sequential images
    # sequential image features: [B*T, C, H, W], F.grid_sample(features, pixels)
    # so the first dimension of pixels should be B*T
    points_2d = points_2d.permute(0, 3, 2, 1, 4, 5).contiguous() # [B, T, N, C, P, 2]
    points_2d = points_2d.view(B*T, N, C, P, 2)
    return points_2d
    
def project_points_to_image(trajs: torch.Tensor, 
                            calibrations: torch.Tensor, 
                            ego_states: torch.Tensor,
                            unit_points: torch.Tensor,
                            normalize: bool = True) -> torch.Tensor:
    """
    Project points to image.
    Args:
        trajs: torch.Tensor[B, N, TrajParamIndex.END_OF_INDEX]
        calibrations: torch.Tensor[B, C, CameraParamIndex.END_OF_INDEX]
        ego_states: torch.Tensor[B, T, EgoStateIndex.END_OF_INDEX]
        unit_points: torch.Tensor[B, N, 3, P] or torch.Tensor[3, P]
    Returns:
        pixels: torch.Tensor[B*T, N, C, P, 2]
    """
    
    # get the transform from local object coordinates to camera frame
    tr_ob_to_camera = get_transform_from_object_to_camera(trajs, calibrations, ego_states)
    tr_ob_to_camera = tr_ob_to_camera.to(unit_points.dtype)
    B, C, N, T, _, _ = tr_ob_to_camera.shape  # [B, C, N, T, 3, 4]
    P = unit_points.shape[-1]
    
    # extract object size and generate points in object coordinates
    dims = trajs[..., TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1].unsqueeze(-1)  # [B, N, 3, 1]
    object_points = unit_points * dims  # [B, N, 3, P]
    
    ones = torch.ones((B, N, 1, P)).to(object_points.device)
    homogeneous_points = torch.cat([object_points, ones], dim=-2)  # [B, N, 4, P]
    
    # transform points to camera frame
    use_einsum = False
    if use_einsum:
        # method1: use einsum
        # print(f"homogeneous_points is contiguous: {homogeneous_points.is_contiguous()}") # True
        cam_points = torch.einsum('bcntij,bnjp->bcntpi', tr_ob_to_camera, homogeneous_points)  # [B, C, N, T, P, 3]
        # print(f"cam_points is contiguous: {cam_points.is_contiguous()}") # False
        cam_points = cam_points.contiguous()
    else:
        # method2: use matmul
        # with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        homogeneous_points = homogeneous_points.unsqueeze(1).unsqueeze(3)  # [B, 1, N, 1, 4, P]
        cam_points = torch.matmul(tr_ob_to_camera, homogeneous_points) # [B, C, N, T, 3, P]
        cam_points = cam_points.permute(0, 1, 2, 3, 5, 4).contiguous() # [B, C, N, T, P, 3]
        # print(f"cam_points is contiguous: {cam_points.is_contiguous()}") # True
    cam_points = cam_points.to(calibrations.dtype) 
    # due to the 16-mixed precision, the cam_points could be optimized to float16
    # in matrix multiplication, the cam_points is float16, but the calibrations is float32
    # so we need to cast the cam_points to float32
    
    # pixels: torch.Tensor[B*T, N, C, P, 2]
    return camera_to_pixel(cam_points, calibrations, normalize)

if __name__ == "__main__":
    B, N, T, C, P = 32, 128, 10, 7, 128
    trajs = torch.randn(B, N, TrajParamIndex.END_OF_INDEX)
    calibrations = torch.randn(B, C, CameraParamIndex.END_OF_INDEX)
    ego_states = torch.randn(B, T, EgoStateIndex.END_OF_INDEX)
    unit_points = torch.randn(3, P)
    dims = trajs[..., TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1].unsqueeze(-1)  # [B, N, 3, 1]
    object_points = unit_points * dims  # [B, N, 3, P]
    homogeneous_points = torch.cat([object_points, torch.ones((B, N, 1, P))], dim=-2)  # [B, N, 4, P]
    
    tr_ob_to_camera = get_transform_from_object_to_camera(trajs, calibrations, ego_states)
    R_ob_to_camera, t_ob_to_camera = tr_ob_to_camera[..., :3, :3], tr_ob_to_camera[..., :3, 3:4]
    
    def einsum_matmul_test(A: torch.Tensor, B: torch.Tensor):
        C = torch.einsum('bcntij,bnjp->bcntpi', A, B)
        C = C.contiguous()
        return C
    
    def matmul_test(A: torch.Tensor, B: torch.Tensor):
        B = B.unsqueeze(1).unsqueeze(3)  # [B, 1, N, 1, 4, P]
        C = torch.matmul(A, B)
        # print(f"C is contiguous: {C.is_contiguous()}") # True
        return C
    
    def matmul_test2(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
        C = C.unsqueeze(1).unsqueeze(3)  # [B, 1, N, 1, 4, P]
        D = torch.matmul(A, C) + B
        return D
    
    R_ob_to_camera = R_ob_to_camera.to(torch.device('cuda:0'))
    t_ob_to_camera = t_ob_to_camera.to(torch.device('cuda:0'))
    object_points = object_points.to(torch.device('cuda:0'))
    
    tr_ob_to_camera = tr_ob_to_camera.to(torch.device('cuda:0'))
    homogeneous_points = homogeneous_points.to(torch.device('cuda:0'))
    
    measure_gpu_latency(einsum_matmul_test, tr_ob_to_camera, homogeneous_points)
    measure_gpu_latency(matmul_test, tr_ob_to_camera, homogeneous_points)
    measure_gpu_latency(matmul_test2, R_ob_to_camera, t_ob_to_camera, object_points)
    
    # GPU Latency of einsum_matmul_test: 41.731201 ms
    # GPU Latency of matmul_test: 4.073152 ms
    # GPU Latency of matmul_test2: 7.103424 ms
    
    measure_average_gpu_latency(einsum_matmul_test, tr_ob_to_camera, homogeneous_points)
    measure_average_gpu_latency(matmul_test, tr_ob_to_camera, homogeneous_points)
    measure_average_gpu_latency(matmul_test2, R_ob_to_camera, t_ob_to_camera, object_points)
    
    # Average GPU Latency of einsum_matmul_test: 1.733967 ms
    # Average GPU Latency of matmul_test: 2.772202 ms
    # Average GPU Latency of matmul_test2: 3.575904 ms
    
    # CONCLUSION:
    # 1. with single run, matmul is faster than einsum
    # 2. with multiple runs, einsum is faster than matmul when einsum has cache
    # 3. use Tr * Homo_points is faster than R * points + t
