import pytest
import torch
import torch.nn.functional as F
from xinnovation.src.components.lightning_module.losses.classification import FocalLoss, MultiClassFocalLoss
from xinnovation.src.core.registry import LOSSES


def test_losses_registry():
    """Test that loss functions can be built from the LOSSES registry."""
    # Setup test data
    pred = torch.tensor([0.7, 0.3, 0.8, 0.1]).unsqueeze(0)
    target = torch.tensor([1.0, 0.0, 1.0, 0.0]).unsqueeze(0)
    
    # Build FocalLoss from registry
    focal_loss_cfg = dict(
        type='FocalLoss',
        alpha=[0.25, 0.25, 0.25, 0.25],
        gamma=2.0
    )
    focal_loss = LOSSES.build(focal_loss_cfg)
    assert isinstance(focal_loss, FocalLoss)
    
    # Test the built loss function
    loss = focal_loss(pred, target)
    assert not torch.isnan(loss)
    
    # Build MultiClassFocalLoss from registry
    multi_pred = torch.tensor([
        [1.0, 0.2, 0.1],
        [0.2, 2.0, 0.3]
    ]) # (B, C) = (2, 3)
    multi_target = torch.tensor([0, 1])
    
    multiclass_loss_cfg = dict(
        type='MultiClassFocalLoss',
        alpha=[0.3, 0.4, 0.3],
        gamma=2.0
    )
    multiclass_loss = LOSSES.build(multiclass_loss_cfg)
    assert isinstance(multiclass_loss, MultiClassFocalLoss)
    
    # # Test the built loss function
    multi_loss = multiclass_loss(multi_pred, multi_target)
    assert not torch.isnan(multi_loss)
