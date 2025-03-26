import os
import sys
import torch
import torch.nn as nn
import pytest
from xinnovation.src.core.registry import (
    DETECTORS, 
    LOSSES,
    OPTIMIZERS,
    SCHEDULERS
)
from lightning.pytorch import Trainer
from xinnovation.src.components.lightning_module import LightningDetector, AdamOptimizer, StepLRScheduler

# Mock implementations for testing
class MockBackbone(nn.Module):
    def __init__(self, depth=50):
        super().__init__()
        self.depth = depth
        self.conv = nn.Conv2d(3, 64, kernel_size=3)
    
    def forward(self, x):
        return {"features": [torch.randn(1, 64, 32, 32)]}
    
    def __str__(self):
        return f"MockBackbone(depth={self.depth})"

class MockNeck(nn.Module):
    def __init__(self, in_channels=None, out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        return torch.randn(1, self.out_channels, 32, 32)
    
    def __str__(self):
        return f"MockNeck(out_channels={self.out_channels})"

# Test model configuration
lightning_module_cfg = {
    "detector": {
        "type": "MockDetector",
    },
    "loss": {
        "type": "MockLoss",
    },
    "optimizer": {
        "type": "AdamOptimizer",
        "lr": 0.001
    },
    "scheduler": {
        "type": "StepLRScheduler",
        "step_size": 10,
        "gamma": 0.1
    }
}

# Register test models
@DETECTORS.register_module()
class MockDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MockBackbone()
        self.neck = MockNeck()
        
    def forward(self, x):
        return {"pred": torch.randn(1, 80, 32, 32)}
        
    def __str__(self):
        return f"MockDetector(backbone={self.backbone}, neck={self.neck})"
            
@LOSSES.register_module()
class MockLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        return torch.tensor(0.5, requires_grad=True)
        
    def __str__(self):
        return "MockLoss()"

@pytest.fixture
def lightning_detector():
    """Create a LightningDetector instance for testing"""
    lightning_module = LightningDetector(lightning_module_cfg)
    lightning_module.configure_optimizers()
    return lightning_module


def test_initialization(lightning_detector):
    """Test if the model initializes correctly"""
    assert lightning_detector is not None
    assert hasattr(lightning_detector, 'detector')
    assert hasattr(lightning_detector, 'criterion')


def test_configure_optimizers(lightning_detector):
    """Test if optimizers and schedulers are configured correctly"""
    optim_dict = lightning_detector.configure_optimizers()
    assert 'optimizer' in optim_dict
    assert 'lr_scheduler' in optim_dict
    assert isinstance(optim_dict['optimizer'], torch.optim.Adam)
    assert isinstance(optim_dict['lr_scheduler'], torch.optim.lr_scheduler.StepLR)


def test_string_representation(lightning_detector):
    """Test the string representation of the model"""
    model_str = str(lightning_detector)
    assert 'LightningDetector' in model_str
    assert 'detector' in model_str
    assert 'criterion' in model_str


def test_optimizer_params(lightning_detector):
    """Test that optimizer has the correct parameters"""
    optim_dict = lightning_detector.configure_optimizers()
    optimizer = optim_dict['optimizer']
    
    # Check that optimizer contains parameters from the model
    param_groups = optimizer.param_groups
    assert len(param_groups) > 0
    assert len(param_groups[0]['params']) > 0
    
    # Verify optimizer hyperparameters
    assert param_groups[0]['lr'] == 0.001


@pytest.mark.parametrize("lr,step_size,gamma", [
    (0.01, 5, 0.5),
    (0.001, 10, 0.1),
    (0.0001, 15, 0.2)
])
def test_custom_hyperparameters(lr, step_size, gamma):
    """Test model with different hyperparameters"""
    
    # Create config with custom hyperparameters
    custom_cfg = lightning_module_cfg.copy()
    custom_cfg['optimizer']['lr'] = lr
    custom_cfg['scheduler']['step_size'] = step_size
    custom_cfg['scheduler']['gamma'] = gamma
    
    # Create model with custom config
    model = LightningDetector(custom_cfg)
    optim_dict = model.configure_optimizers()
    
    # Check that hyperparameters were set correctly
    optimizer = optim_dict['optimizer']
    scheduler = optim_dict['lr_scheduler']
    
    assert optimizer.param_groups[0]['lr'] == lr
    assert scheduler.step_size == step_size
    assert scheduler.gamma == gamma


def test_detector_assignment(lightning_detector):
    """Test that the detector is correctly assigned and accessible"""
    # Test detector is accessible
    detector = lightning_detector.detector
    assert detector is not None
    assert isinstance(detector, MockDetector)
    
    # Verify detector components
    assert hasattr(detector, 'backbone')
    assert hasattr(detector, 'neck')
    assert isinstance(detector.backbone, MockBackbone)
    assert isinstance(detector.neck, MockNeck)


def test_criterion_assignment(lightning_detector):
    """Test that the loss criterion is correctly assigned and accessible"""
    # Test criterion is accessible
    criterion = lightning_detector.criterion
    assert criterion is not None
    assert isinstance(criterion, MockLoss)


if __name__ == "__main__":
    print("Running LightningDetector tests...")
    
    try:
        
        # Create model using Dict directly
        model = LightningDetector(lightning_module_cfg)
        optim_dict = model.configure_optimizers()
        print("\nBasic test results:")
        print("-" * 50)
        print(f"Model initialized: {model is not None}")
        print(f"Optimizer configured: {'optimizer' in optim_dict}")
        print(f"LR Scheduler configured: {'lr_scheduler' in optim_dict}")
        print("\nModel structure:")
        print("-" * 50)
        print(model)
        
        print("\nAll manual tests passed!")
        
        # Run pytest programmatically
        pytest.main(["-xvs", __file__])
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc() 