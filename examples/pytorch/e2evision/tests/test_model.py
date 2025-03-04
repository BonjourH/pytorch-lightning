import torch
import pytest
from typing import Dict, List
import numpy as np

from base import SourceCameraId
from models import (
    ImageFeatureExtractor,
    TemporalAttentionFusion,
    TrajectoryRefinementLayer,
    E2EPerceptionNet
)

@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    B, T, C, H, W = 2, 10, 3, 224, 224
    
    images = {
        SourceCameraId.FRONT_CENTER_CAMERA: torch.randn(B, T, C, H, W),
        SourceCameraId.FRONT_LEFT_CAMERA: torch.randn(B, T, C, H, W),
        SourceCameraId.FRONT_RIGHT_CAMERA: torch.randn(B, T, C, H, W)
    }
    
    calibrations = {
        camera_id: torch.randn(B, 21) for camera_id in images.keys()
    }
    
    ego_states = torch.randn(B, T, 9)
    
    return {
        'images': images,
        'calibrations': calibrations,
        'ego_states': ego_states
    }

def test_image_feature_extractor():
    """Test ImageFeatureExtractor."""
    model = ImageFeatureExtractor(out_channels=256)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    
    assert out.shape[1] == 256, "Output channel dimension mismatch"
    assert out.shape[2] == x.shape[2] // 32, "Feature map height mismatch"
    assert out.shape[3] == x.shape[3] // 32, "Feature map width mismatch"

def test_temporal_attention_fusion():
    """Test TemporalAttentionFusion."""
    B, T, C, H, W = 2, 10, 256, 7, 14
    model = TemporalAttentionFusion(feature_dim=C)
    x = torch.randn(B, T, C, H, W)
    out = model(x)
    
    assert out.shape == (B, C, H, W), "Output shape mismatch"

def test_trajectory_refinement():
    """Test TrajectoryRefinementLayer."""
    B, N, D = 2, 100, 512
    model = TrajectoryRefinementLayer(hidden_dim=D)
    queries = torch.randn(B, N, D)
    features = torch.randn(B, D, 7, 14)
    out = model(queries, features)
    
    assert out.shape == queries.shape, "Output shape mismatch"

def test_e2e_perception_net(sample_batch):
    """Test E2EPerceptionNet."""
    camera_ids = list(sample_batch['images'].keys())
    model = E2EPerceptionNet(
        camera_ids=camera_ids,
        feature_dim=256,
        num_queries=100,
        num_decoder_layers=6
    )
    
    outputs = model(sample_batch)
    
    # Check output format
    assert isinstance(outputs, list), "Output should be a list"
    assert len(outputs) == model.num_decoder_layers + 1, "Wrong number of outputs"
    
    # Check final prediction
    final_pred = outputs[-1]
    assert 'traj_params' in final_pred, "Missing trajectory parameters"
    assert 'type_logits' in final_pred, "Missing type logits"
    
    B = sample_batch['images'][camera_ids[0]].shape[0]
    assert final_pred['traj_params'].shape[0] == B, "Batch size mismatch"
    assert final_pred['traj_params'].shape[1] == model.num_queries, "Number of queries mismatch"

def test_model_gradient_flow(sample_batch):
    """Test gradient flow through the model."""
    camera_ids = list(sample_batch['images'].keys())
    model = E2EPerceptionNet(
        camera_ids=camera_ids,
        feature_dim=256,
        num_queries=100,
        num_decoder_layers=6
    )
    
    # Forward pass
    outputs = model(sample_batch)
    
    # Create dummy loss
    loss = outputs[-1]['traj_params'].sum()
    
    # Backward pass
    loss.backward()
    
    # Check if gradients exist
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

def test_model_device_transfer(sample_batch):
    """Test model device transfer."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    camera_ids = list(sample_batch['images'].keys())
    model = E2EPerceptionNet(
        camera_ids=camera_ids,
        feature_dim=256,
        num_queries=100,
        num_decoder_layers=6
    ).cuda()
    
    # Move batch to GPU
    cuda_batch = {
        'images': {k: v.cuda() for k, v in sample_batch['images'].items()},
        'calibrations': {k: v.cuda() for k, v in sample_batch['calibrations'].items()},
        'ego_states': sample_batch['ego_states'].cuda()
    }
    
    # Forward pass
    outputs = model(cuda_batch)
    
    # Check if outputs are on GPU
    assert outputs[-1]['traj_params'].device.type == 'cuda', "Output not on GPU"

def test_model_jit_compatibility(sample_batch):
    """Test model TorchScript compatibility."""
    camera_ids = list(sample_batch['images'].keys())
    model = E2EPerceptionNet(
        camera_ids=camera_ids,
        feature_dim=256,
        num_queries=100,
        num_decoder_layers=6
    )
    
    # Try to JIT compile
    try:
        jit_model = torch.jit.script(model)
    except Exception as e:
        pytest.fail(f"Failed to JIT compile model: {e}")
    
    # Test forward pass with JIT model
    outputs = jit_model(sample_batch)
    assert isinstance(outputs, list), "JIT model output format mismatch" 