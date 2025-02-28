#!/bin/bash

# Set the Python path to include the project root
export PYTHONPATH="$PYTHONPATH:$(pwd)/../../.."

# Set test data paths
TEST_CLIP="/home/xinjian/Code/VAutoLabelerCore/labeling_info/1_20231219T122348_pdb-l4e-c0011_0_0to8"

# Create a temporary test script
TMP_SCRIPT=$(mktemp)
cat > "$TMP_SCRIPT" << 'EOF'
from data import MultiFrameDataset
from base import SourceCameraId
from torch.utils.data import DataLoader

# Test configuration
clip_dirs = [
    '/home/xinjian/Code/VAutoLabelerCore/labeling_info/1_20231219T122348_pdb-l4e-c0011_0_0to8'
]

camera_ids = [
    SourceCameraId.FRONT_CENTER_CAMERA,
    SourceCameraId.FRONT_LEFT_CAMERA,
    SourceCameraId.FRONT_RIGHT_CAMERA,
    SourceCameraId.SIDE_LEFT_CAMERA,
    SourceCameraId.SIDE_RIGHT_CAMERA,
    SourceCameraId.REAR_LEFT_CAMERA,
    SourceCameraId.REAR_RIGHT_CAMERA
]

print("\n=== Testing Dataset Initialization ===")
dataset = MultiFrameDataset(
    clip_dirs=clip_dirs,
    camera_ids=camera_ids,
    sequence_length=10
)

print("\n=== Testing Data Loading ===")
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=2
)

print("\n=== Testing Batch Processing ===")
for batch in dataloader:
    print("\nBatch contents:")
    print(f"Number of cameras: {len(batch['images'])}")
    
    print("\nImage shapes:")
    for camera_id, images in batch['images'].items():
        print(f"Camera {camera_id.name}: {images.shape}")
    
    print("\nEgo state info:")
    print(f"Number of ego states: {len(batch['ego_states'])}")
    print(f"First ego state keys: {batch['ego_states'][0].keys()}")
    
    print("\nObject info:")
    print(f"Number of frames: {len(batch['objects_data'])}")
    if len(batch['objects_data']) > 0:
        frame_data = batch['objects_data'][0]
        print(f"First frame object data:")
        print(f"  Number of objects: {len(frame_data['ids'])}")
        if len(frame_data['ids']) > 0:
            print(f"  Types shape: {frame_data['types'].shape}")
            print(f"  Positions shape: {frame_data['positions'].shape}")
            print(f"  Dimensions shape: {frame_data['dimensions'].shape}")
            print(f"  Yaws shape: {frame_data['yaws'].shape}")
            print(f"  Velocities shape: {frame_data['velocities'].shape}")
    
    print("\nCalibration info:")
    for camera_id, calib in batch['calibrations'].items():
        print(f"\nCamera {camera_id.name}:")
        print(f"  Camera type: {calib['camera_type'].name}")
        print(f"  Intrinsic shape: {calib['intrinsic'].shape}")
        print(f"  Extrinsic shape: {calib['extrinsic'].shape}")
        if calib['distortion'].numel() > 0:  # Check if distortion tensor is not empty
            print(f"  Distortion shape: {calib['distortion'].shape}")
    
    # Only process one batch
    break

print("\n=== Test Completed Successfully ===")
EOF

# Run the test script
echo "Running data.py tests..."
python "$TMP_SCRIPT"

# Clean up
rm "$TMP_SCRIPT" 