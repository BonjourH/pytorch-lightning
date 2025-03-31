import pytest
import torch
from xinnovation.examples.detector4D.sparse4d_loss import Sparse4DLossWithDAC, compute_has_object_cls_cost_matrix
from xinnovation.src.core import TrajParamIndex

@pytest.fixture
def test_sparse4d_loss():
    pass

def test_compute_cls_cost_matrix():
    """Test the classification cost matrix computation with M=3 ground truths and N=2 predictions."""
    # Test case 1: Basic shape and non-negativity
    B, M, N = 1, 3, 2
    gt_trajs = torch.zeros(B, M, TrajParamIndex.END_OF_INDEX)
    pred_trajs = torch.zeros(B, N, TrajParamIndex.END_OF_INDEX)
    
    # Set objectness scores for ground truths
    gt_trajs[0, 0, TrajParamIndex.HAS_OBJECT] = 1.0    # Positive
    gt_trajs[0, 1, TrajParamIndex.HAS_OBJECT] = 0.0    # Negative
    gt_trajs[0, 2, TrajParamIndex.HAS_OBJECT] = 1.0    # Positive
    
    # Set objectness scores for predictions
    pred_trajs[0, 0, TrajParamIndex.HAS_OBJECT] = 2.0  # High confidence positive (logit)
    pred_trajs[0, 1, TrajParamIndex.HAS_OBJECT] = -2.0  # High confidence negative (logit)
    
    cost_matrix = compute_has_object_cls_cost_matrix(gt_trajs, pred_trajs)
    
    # Check shape
    assert cost_matrix.shape == (B, M, N)
    
    # Check that all costs are non-negative
    assert torch.all(cost_matrix >= 0)
    
    # Test case 2: Batch size > 1 with different patterns
    B, M, N = 2, 3, 2
    gt_trajs = torch.zeros(B, M, TrajParamIndex.END_OF_INDEX)
    pred_trajs = torch.zeros(B, N, TrajParamIndex.END_OF_INDEX)
    
    # First batch: High confidence predictions
    gt_trajs[0, 0, TrajParamIndex.HAS_OBJECT] = 1.0    # Positive
    gt_trajs[0, 1, TrajParamIndex.HAS_OBJECT] = 0.0    # Negative
    gt_trajs[0, 2, TrajParamIndex.HAS_OBJECT] = 1.0    # Positive
    pred_trajs[0, 0, TrajParamIndex.HAS_OBJECT] = 2.0  # High confidence positive
    pred_trajs[0, 1, TrajParamIndex.HAS_OBJECT] = -2.0  # High confidence negative
    
    # Second batch: Different pattern
    gt_trajs[1, 0, TrajParamIndex.HAS_OBJECT] = 0.0    # Negative
    gt_trajs[1, 1, TrajParamIndex.HAS_OBJECT] = 1.0    # Positive
    gt_trajs[1, 2, TrajParamIndex.HAS_OBJECT] = 0.0    # Negative
    pred_trajs[1, 0, TrajParamIndex.HAS_OBJECT] = -2.0  # High confidence negative
    pred_trajs[1, 1, TrajParamIndex.HAS_OBJECT] = 2.0   # High confidence positive
    
    cost_matrix = compute_has_object_cls_cost_matrix(gt_trajs, pred_trajs)
    
    # Check shape
    assert cost_matrix.shape == (B, M, N)
    
    # Check that costs are non-negative
    assert torch.all(cost_matrix >= 0)
    
    # Test case 3: Edge cases with very confident predictions
    B, M, N = 1, 3, 2
    gt_trajs = torch.zeros(B, M, TrajParamIndex.END_OF_INDEX)
    pred_trajs = torch.zeros(B, N, TrajParamIndex.END_OF_INDEX)
    
    # Test with very high confidence predictions
    gt_trajs[0, 0, TrajParamIndex.HAS_OBJECT] = 1.0    # Positive
    gt_trajs[0, 1, TrajParamIndex.HAS_OBJECT] = 0.0    # Negative
    gt_trajs[0, 2, TrajParamIndex.HAS_OBJECT] = 1.0    # Positive
    pred_trajs[0, 0, TrajParamIndex.HAS_OBJECT] = 5.0  # Very high confidence positive (logit)
    pred_trajs[0, 1, TrajParamIndex.HAS_OBJECT] = -5.0  # Very high confidence negative (logit)
    
    confident_cost_matrix = compute_has_object_cls_cost_matrix(gt_trajs, pred_trajs)
    
    # Check that correct high confidence predictions have low cost
    assert confident_cost_matrix[0, 0, 0] < 0.1  # Positive ground truth matching positive prediction
    assert confident_cost_matrix[0, 1, 1] < 0.1  # Negative ground truth matching negative prediction
    
    # Test case 4: Uncertain predictions
    B, M, N = 1, 3, 2
    gt_trajs = torch.zeros(B, M, TrajParamIndex.END_OF_INDEX)
    pred_trajs = torch.zeros(B, N, TrajParamIndex.END_OF_INDEX)
    
    # Set uncertain predictions
    gt_trajs[0, 0, TrajParamIndex.HAS_OBJECT] = 1.0    # Positive
    gt_trajs[0, 1, TrajParamIndex.HAS_OBJECT] = 0.0    # Negative
    gt_trajs[0, 2, TrajParamIndex.HAS_OBJECT] = 1.0    # Positive
    pred_trajs[0, 0, TrajParamIndex.HAS_OBJECT] = 0.5  # Slightly positive
    pred_trajs[0, 1, TrajParamIndex.HAS_OBJECT] = -0.5  # Slightly negative
    
    uncertain_cost_matrix = compute_has_object_cls_cost_matrix(gt_trajs, pred_trajs)
    
    # Check that uncertain predictions have higher costs than confident ones
    assert uncertain_cost_matrix[0, 0, 0] > confident_cost_matrix[0, 0, 0]  # Uncertain positive should have higher cost than confident positive
    assert uncertain_cost_matrix[0, 1, 1] > confident_cost_matrix[0, 1, 1]  # Uncertain negative should have higher cost than confident negative

