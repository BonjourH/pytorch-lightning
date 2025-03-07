#!/bin/bash

# Set the Python path to include the project root
export PYTHONPATH="$PYTHONPATH:$(pwd)/../../.."

# Set test data paths
TEST_CLIP="/home/xinjian/Code/VAutoLabelerCore/labeling_info/1_20231219T122348_pdb-l4e-c0011_0_0to8"

# Create a temporary test script
TMP_SCRIPT=$(mktemp)
cat > "$TMP_SCRIPT" << 'EOF'
from e2e_dataset.dataset import MultiFrameDataset, custom_collate_fn
from base import SourceCameraId, CameraType, CameraParamIndex, TrajParamIndex, ObjectType, tensor_to_object_type
from torch.utils.data import DataLoader
from configs.config import DataConfig, CameraGroupConfig
import torch

# Test configuration
clip_dirs = [
    '/home/xinjian/Code/VAutoLabelerCore/labeling_info/1_20231219T122348_pdb-l4e-c0011_0_0to8'
]

# Create a DataConfig object
data_config = DataConfig()
data_config.sequence_length = 10
data_config.camera_ids = [
    SourceCameraId.FRONT_CENTER_CAMERA,
    SourceCameraId.FRONT_LEFT_CAMERA,
    SourceCameraId.FRONT_RIGHT_CAMERA,
    SourceCameraId.SIDE_LEFT_CAMERA,
    SourceCameraId.SIDE_RIGHT_CAMERA,
    SourceCameraId.REAR_LEFT_CAMERA,
    SourceCameraId.REAR_RIGHT_CAMERA
]
data_config.camera_groups = [
    CameraGroupConfig.front_stereo_camera_group(),
    CameraGroupConfig.short_focal_length_camera_group(),
    CameraGroupConfig.rear_camera_group()
]

print("\n=== Testing Dataset Initialization ===")
dataset = MultiFrameDataset(
    clip_dirs=clip_dirs,
    config=data_config
)

print("\n=== Testing Data Loading ===")
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=2,
    collate_fn=custom_collate_fn
)

print("\n=== Testing Batch Processing ===")
for batch in dataloader:
    print("\nBatch contents:")
    print(f"Number of camera groups: {len(batch['images'])}")
    
    print("\nImage shapes:")
    for camera_group_name, images in batch['images'].items():
        print(f"Camera group {camera_group_name}: {images.shape}")
    
    print("\nEgo state info:")
    print(f"Shape of ego states tensor: {batch['ego_states'].shape}")
    
    print("\nTrajectory info:")
    print(f"Shape of trajectories tensor: {batch['trajs'].shape}")
    
    # Count valid objects (those with HAS_OBJECT=1)
    valid_traj_counts = (batch['trajs'][:, :, TrajParamIndex.HAS_OBJECT] > 0.5).sum(dim=1)
    print(f"Valid trajectories per sample: {valid_traj_counts}")
    
    # Print info about the first valid trajectory in the first sample
    if valid_traj_counts[0] > 0:
        # Find first valid trajectory
        first_valid_idx = torch.nonzero(batch['trajs'][0, :, TrajParamIndex.HAS_OBJECT] > 0.5)[0].item()
        first_traj = batch['trajs'][0, first_valid_idx]
        
        print(f"\nFirst valid trajectory details:")
        print(f"  Position: ({first_traj[TrajParamIndex.X]:.2f}, {first_traj[TrajParamIndex.Y]:.2f}, {first_traj[TrajParamIndex.Z]:.2f})")
        print(f"  Velocity: ({first_traj[TrajParamIndex.VX]:.2f}, {first_traj[TrajParamIndex.VY]:.2f})")
        print(f"  Dimensions: {first_traj[TrajParamIndex.LENGTH]:.2f} x {first_traj[TrajParamIndex.WIDTH]:.2f} x {first_traj[TrajParamIndex.HEIGHT]:.2f}")
        print(f"  Yaw: {first_traj[TrajParamIndex.YAW]:.2f}")
        print(f"  Object type: {tensor_to_object_type(first_traj)}")
    
    print("\nCalibration info:")
    for camera_id, calib in batch['calibrations'].items():
        print(f"\nCamera {camera_id.name}:")
        print(f"  Camera parameters shape: {calib.shape}")
        print(f"  Camera type: {CameraType(int(calib[0, CameraParamIndex.CAMERA_TYPE].item()))}")
    
    # Only process one batch
    break

print("\n=== Test Completed Successfully ===")
EOF

# Run the test script
echo "Running dataset tests..."
python "$TMP_SCRIPT"

# Clean up
rm "$TMP_SCRIPT" 