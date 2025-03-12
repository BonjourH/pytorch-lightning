import torch
from base import TrajParamIndex, EgoStateIndex, CameraParamIndex, CameraType
from utils.latency_utils import measure_gpu_latency, measure_average_gpu_latency

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
    
    yaw = trajs[..., TrajParamIndex.YAW].unsqueeze(-1)  # [B, N, 1]
    
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
    
    # If speed is sufficient, use velocity direction; otherwise use provided yaw
    yaw_t = torch.where(speed_t > 0.2, torch.atan2(vel_y_t, vel_x_t), yaw)
    
    # now we have pos_x_t, pos_y_t, pos_z_t, yaw_t, which is the Transform from object to current ego frame
    # shape: [B, N, T]
    
    # 2. get transformation T_ego_temporal_to_ego_previous from current ego frame to previous ego frame
    # ego states is the transform from current ego frame to previous ego frame
    ego_yaw = ego_states[..., EgoStateIndex.YAW].unsqueeze(1)  # [B, 1, T]
    ego_x = ego_states[..., EgoStateIndex.X].unsqueeze(1)  # [B, 1, T]
    ego_y = ego_states[..., EgoStateIndex.Y].unsqueeze(1)  # [B, 1, T]
    
    # 3. get transformation T_ob_to_ego_previous from local object coordinates to previous ego frame
    cos_yaw = torch.cos(yaw_t)
    sin_yaw = torch.sin(yaw_t)
    x_rotated = pos_x_t * cos_yaw - pos_y_t * sin_yaw + ego_x # [B, N, T]
    y_rotated = pos_x_t * sin_yaw + pos_y_t * cos_yaw + ego_y # [B, N, T]
    
    yaw_to_previous_ego = yaw_t + ego_yaw # [B, N, T]
    
    zeros = torch.zeros_like(cos_yaw)
    ones = torch.ones_like(cos_yaw)
    
    R_ob_to_ego_previous = torch.stack([
        cos_yaw, -sin_yaw, zeros,
        sin_yaw, cos_yaw, zeros,
        zeros, zeros, ones
    ], dim=-1).reshape(B, N, T, 3, 3)
    
    t_ob_to_ego_previous = torch.stack([
        x_rotated,
        y_rotated,
        pos_z_t * torch.ones_like(x_rotated)
    ], dim=-1).reshape(B, N, T, 3, 1)
    
    # 4. calculate the dynamic camera extrinsic parameters with pitch correction
    pitch_correction = ego_states[..., EgoStateIndex.PITCH_CORRECTION] # [B, T]
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
        pixels: torch.Tensor[B, C, N, T, P, 2]
        behind_camera: torch.Tensor[B, C, N, T, P, 1]
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
    
    # Get image dimensions
    img_width = calib_params[..., CameraParamIndex.IMAGE_WIDTH].unsqueeze(2)  # [B, C, 1]
    img_height = calib_params[..., CameraParamIndex.IMAGE_HEIGHT].unsqueeze(2)  # [B, C, 1]
    
    # Check if points are behind the camera
    x_cam = cam_points[..., 0] # [B, C, NTP]
    y_cam = cam_points[..., 1] # [B, C, NTP]
    z_cam = cam_points[..., 2] # [B, C, NTP]
    
    # Handle division by zero
    z_cam = torch.where(z_cam == 0, torch.ones_like(z_cam) * 1e-10, z_cam)
    
    # Normalize coordinates
    x_normalized = x_cam / z_cam 
    y_normalized = y_cam / z_cam
    
    # calculate fisheye distorted coordinates
    fisheye_r = torch.sqrt(x_normalized * x_normalized + y_normalized * y_normalized)
    fisheye_r = torch.where(fisheye_r == 0, torch.ones_like(fisheye_r) * 1e-10, fisheye_r)
    
    fisheye_theta = torch.atan(fisheye_r)
    fisheye_theta2 = fisheye_theta * fisheye_theta
    fisheye_theta4 = fisheye_theta2 * fisheye_theta2
    fisheye_theta6 = fisheye_theta4 * fisheye_theta2
    fisheye_theta8 = fisheye_theta4 * fisheye_theta4
    
    fisheye_theta_d = fisheye_theta * (1 + k1 * fisheye_theta2 + k2 * fisheye_theta4 + k3 * fisheye_theta6 + k4 * fisheye_theta8)
    scaling = torch.where(fisheye_r > 0, fisheye_theta_d / fisheye_r, torch.ones_like(fisheye_r))
    
    x_distorted_fisheye = x_normalized * scaling
    y_distorted_fisheye = y_normalized * scaling
    
    # calculate general distorted coordinates
    general_r2 = x_normalized * x_normalized + y_normalized * y_normalized
    general_r4 = general_r2 * general_r2
    general_r6 = general_r4 * general_r2
    
    radial = 1 + k1 * general_r2 + k2 * general_r4 + k3 * general_r6
    
    dx = 2 * p1 * x_normalized * y_normalized + p2 * (general_r2 + 2 * x_normalized * x_normalized)
    dy = p1 * (general_r2 + 2 * y_normalized * y_normalized) + 2 * p2 * x_normalized * y_normalized
    
    x_distorted_general = x_normalized * radial + dx
    y_distorted_general = y_normalized * radial + dy
    
    # calculate pinhole distorted coordinates
    # no need to calculate
    # x_distorted_pinhole = x_normalized
    # y_distorted_pinhole = y_normalized
    
    # combine all distorted coordinates
    fish_eye_mask = (camera_type == CameraType.FISHEYE).unsqueeze(2)
    general_distort_mask = (camera_type == CameraType.GENERAL_DISTORT).unsqueeze(2)
    # pinhole_mask = (camera_type == CameraType.PINHOLE).unsqueeze(2)
    
    x_distorted = torch.where(fish_eye_mask, x_distorted_fisheye, x_distorted_general)
    y_distorted = torch.where(fish_eye_mask, y_distorted_fisheye, y_distorted_general)
    
    # apply camera matrix
    x_pixel = fx * x_distorted + cx
    y_pixel = fy * y_distorted + cy
    
    # normalize to [0, 1] for consistency with the visibility check in gather_point_features
    if normalize:
        x_norm = x_pixel / img_width
        y_norm = y_pixel / img_height
    else:
        x_norm = x_pixel
        y_norm = y_pixel
    
    points_2d = torch.stack([x_norm, y_norm], dim=-1) # [B, C, NTP, 2]
    behind_camera = (z_cam <= 0)  # [B, C, NTP]
    points_2d[behind_camera] = -2.0
    points_2d = points_2d.view(B, C, N, T, P, 2)
    behind_camera = behind_camera.view(B, C, N, T, P, 1)
    return points_2d, behind_camera
    
def project_points_to_image(trajs: torch.Tensor, 
                            calibrations: torch.Tensor, 
                            ego_states: torch.Tensor,
                            unit_points: torch.Tensor) -> torch.Tensor:
    """
    Project points to image.
    Args:
        trajs: torch.Tensor[B, N, TrajParamIndex.END_OF_INDEX]
        calibrations: torch.Tensor[B, C, CameraParamIndex.END_OF_INDEX]
        ego_states: torch.Tensor[B, T, EgoStateIndex.END_OF_INDEX]
        unit_points: torch.Tensor[3, P]
    Returns:
        torch.Tensor[B, C, N, T, P, 2]
    """
    
    # get the transform from local object coordinates to camera frame
    tr_ob_to_camera = get_transform_from_object_to_camera(trajs, calibrations, ego_states)
    tr_ob_to_camera = tr_ob_to_camera.to(unit_points.dtype)
    B, C, N, T, _, _ = tr_ob_to_camera.shape  # [B, C, N, T, 3, 4]
    P = unit_points.shape[1]
    
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
    
    pixels, behind_camera = camera_to_pixel(cam_points, calibrations)
    return pixels, behind_camera


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
