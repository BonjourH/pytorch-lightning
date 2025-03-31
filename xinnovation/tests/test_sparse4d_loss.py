import pytest
import torch
from xinnovation.examples.detector4D.sparse4d_loss import Sparse4DLossWithDAC, compute_has_object_cls_cost_matrix
from xinnovation.src.core import TrajParamIndex

@pytest.fixture
def test_sparse4d_loss():
    pass

def test_compute_cls_cost_matrix():
    """Test the classification cost matrix computation with N=2 predictions and M=3 ground truths."""
    # Test case 1: Basic shape and non-negativity
    B, N, M = 1, 2, 3
    pred_trajs = torch.zeros(B, N, TrajParamIndex.END_OF_INDEX)
    gt_trajs = torch.zeros(B, M, TrajParamIndex.END_OF_INDEX)
    
    # Set objectness scores for predictions
    pred_trajs[0, 0, TrajParamIndex.HAS_OBJECT] = 2.0  # High confidence positive (logit)
    pred_trajs[0, 1, TrajParamIndex.HAS_OBJECT] = -2.0  # High confidence negative (logit)
    
    # Set objectness scores for ground truths
    gt_trajs[0, 0, TrajParamIndex.HAS_OBJECT] = 1.0    # Positive
    gt_trajs[0, 1, TrajParamIndex.HAS_OBJECT] = 0.0    # Negative
    gt_trajs[0, 2, TrajParamIndex.HAS_OBJECT] = 1.0    # Positive
    
    cost_matrix = compute_has_object_cls_cost_matrix(pred_trajs, gt_trajs)
    
    # Check shape
    assert cost_matrix.shape == (B, N, M)
    
    # Check that all costs are non-negative
    assert torch.all(cost_matrix >= 0)
    
    # Test case 2: Batch size > 1 with different patterns
    B, N, M = 2, 2, 3
    pred_trajs = torch.zeros(B, N, TrajParamIndex.END_OF_INDEX)
    gt_trajs = torch.zeros(B, M, TrajParamIndex.END_OF_INDEX)
    
    # First batch: High confidence predictions
    pred_trajs[0, 0, TrajParamIndex.HAS_OBJECT] = 2.0  # High confidence positive
    pred_trajs[0, 1, TrajParamIndex.HAS_OBJECT] = -2.0  # High confidence negative
    gt_trajs[0, 0, TrajParamIndex.HAS_OBJECT] = 1.0    # Positive
    gt_trajs[0, 1, TrajParamIndex.HAS_OBJECT] = 0.0    # Negative
    gt_trajs[0, 2, TrajParamIndex.HAS_OBJECT] = 1.0    # Positive
    
    # Second batch: Different pattern
    pred_trajs[1, 0, TrajParamIndex.HAS_OBJECT] = -2.0  # High confidence negative
    pred_trajs[1, 1, TrajParamIndex.HAS_OBJECT] = 2.0   # High confidence positive
    gt_trajs[1, 0, TrajParamIndex.HAS_OBJECT] = 0.0    # Negative
    gt_trajs[1, 1, TrajParamIndex.HAS_OBJECT] = 1.0    # Positive
    gt_trajs[1, 2, TrajParamIndex.HAS_OBJECT] = 0.0    # Negative
    
    cost_matrix = compute_has_object_cls_cost_matrix(pred_trajs, gt_trajs)
    
    # Check shape
    assert cost_matrix.shape == (B, N, M)
    
    # Check that costs are non-negative
    assert torch.all(cost_matrix >= 0)
    
    # Test case 3: Edge cases with very confident predictions
    B, N, M = 1, 2, 3
    pred_trajs = torch.zeros(B, N, TrajParamIndex.END_OF_INDEX)
    gt_trajs = torch.zeros(B, M, TrajParamIndex.END_OF_INDEX)
    
    # Test with very high confidence predictions
    pred_trajs[0, 0, TrajParamIndex.HAS_OBJECT] = 5.0  # Very high confidence positive (logit)
    pred_trajs[0, 1, TrajParamIndex.HAS_OBJECT] = -5.0  # Very high confidence negative (logit)
    gt_trajs[0, 0, TrajParamIndex.HAS_OBJECT] = 1.0    # Positive
    gt_trajs[0, 1, TrajParamIndex.HAS_OBJECT] = 0.0    # Negative
    gt_trajs[0, 2, TrajParamIndex.HAS_OBJECT] = 1.0    # Positive
    
    confident_cost_matrix = compute_has_object_cls_cost_matrix(pred_trajs, gt_trajs)
    
    # Check that correct high confidence predictions have low cost
    assert confident_cost_matrix[0, 0, 0] < 0.1  # Positive prediction matching positive ground truth
    assert confident_cost_matrix[0, 1, 1] < 0.1  # Negative prediction matching negative ground truth
    
    # Test case 4: Uncertain predictions
    B, N, M = 1, 2, 3
    pred_trajs = torch.zeros(B, N, TrajParamIndex.END_OF_INDEX)
    gt_trajs = torch.zeros(B, M, TrajParamIndex.END_OF_INDEX)
    
    # Set uncertain predictions
    pred_trajs[0, 0, TrajParamIndex.HAS_OBJECT] = 0.5  # Slightly positive
    pred_trajs[0, 1, TrajParamIndex.HAS_OBJECT] = -0.5  # Slightly negative
    gt_trajs[0, 0, TrajParamIndex.HAS_OBJECT] = 1.0    # Positive
    gt_trajs[0, 1, TrajParamIndex.HAS_OBJECT] = 0.0    # Negative
    gt_trajs[0, 2, TrajParamIndex.HAS_OBJECT] = 1.0    # Positive
    
    uncertain_cost_matrix = compute_has_object_cls_cost_matrix(pred_trajs, gt_trajs)
    
    # Check that uncertain predictions have higher costs than confident ones
    assert uncertain_cost_matrix[0, 0, 0] > confident_cost_matrix[0, 0, 0]  # Uncertain positive should have higher cost than confident positive
    assert uncertain_cost_matrix[0, 1, 1] > confident_cost_matrix[0, 1, 1]  # Uncertain negative should have higher cost than confident negative

