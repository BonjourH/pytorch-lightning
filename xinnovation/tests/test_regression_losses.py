import pytest
import torch
from xinnovation.src.components.lightning_module.losses.regression import SmoothL1Loss, IoULoss
from xinnovation.src.core.registry import LOSSES


def test_smooth_l1_loss():
    """Test SmoothL1Loss functionality."""
    # Setup test data
    pred = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    target = torch.tensor([[1.2, 1.8, 3.1, 4.2], [4.9, 6.2, 7.1, 7.8]])
    
    # Test with default parameters
    loss_fn = SmoothL1Loss()
    loss = loss_fn(pred, target)
    assert loss.shape == torch.Size([])  # Scalar output with mean reduction
    assert loss.item() > 0  # Loss should be positive
    
    # Test with different beta
    loss_fn_beta = SmoothL1Loss(beta=0.5)
    loss_beta = loss_fn_beta(pred, target)
    assert not torch.isnan(loss_beta)
    
    # Test different reduction modes
    loss_none = SmoothL1Loss(reduction='none')(pred, target)
    assert loss_none.shape == pred.shape
    
    loss_sum = SmoothL1Loss(reduction='sum')(pred, target)
    assert loss_sum.shape == torch.Size([])


def test_iou_loss():
    """Test IoULoss with different modes."""
    # Setup test data - format is [x1, y1, x2, y2]
    pred = torch.tensor([
        [10.0, 10.0, 20.0, 20.0],  # box 1
        [30.0, 30.0, 40.0, 50.0]   # box 2
    ])
    target = torch.tensor([
        [12.0, 8.0, 22.0, 18.0],   # box 1 target
        [28.0, 32.0, 38.0, 48.0]   # box 2 target
    ])
    
    # Test different IoU loss modes
    modes = ['iou', 'giou', 'diou', 'ciou']
    for mode in modes:
        loss_fn = IoULoss(mode=mode)
        loss = loss_fn(pred, target)
        assert loss.shape == torch.Size([])
        assert 0 <= loss.item() <= 2.0  # IoU-based losses should be between 0 and 2
    
    # Test with batch dimension
    pred_batch = pred.unsqueeze(0).repeat(3, 1, 1)  # [3, 2, 4]
    target_batch = target.unsqueeze(0).repeat(3, 1, 1)  # [3, 2, 4]
    
    loss_batch = IoULoss()(pred_batch, target_batch)
    assert loss_batch.shape == torch.Size([])
    
    # Test reduction='none'
    loss_none = IoULoss(reduction='none')(pred, target)
    assert loss_none.shape == torch.Size([2])  # One loss per box


def test_losses_registry():
    """Test that loss functions can be built from the LOSSES registry."""
    # SmoothL1Loss
    smooth_l1_cfg = dict(
        type='SmoothL1Loss',
        beta=0.5,
        loss_weight=1.0
    )
    smooth_l1 = LOSSES.build(smooth_l1_cfg)
    assert isinstance(smooth_l1, SmoothL1Loss)
    
    # IoULoss
    iou_cfg = dict(
        type='IoULoss',
        mode='giou',
        reduction='mean'
    )
    iou_loss = LOSSES.build(iou_cfg)
    assert isinstance(iou_loss, IoULoss)
    
    # Test with sample inputs
    pred_bbox = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    target_bbox = torch.tensor([[12.0, 8.0, 22.0, 18.0]])
    
    loss = iou_loss(pred_bbox, target_bbox)
    assert not torch.isnan(loss) 