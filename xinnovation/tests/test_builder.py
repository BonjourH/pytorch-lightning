import pytest
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from xinnovation.src.core.builders import (
    build_from_cfg, build_model, build_dataset, build_trainer, 
    build_optimizer, build_scheduler
)
from xinnovation.src.core.registry import (
    LIGHTNING_MODULE, DATA, TRAINER, OPTIMIZERS, SCHEDULERS
)
from xinnovation.src.core.config import Config


# Mock components for testing
class MockModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=3)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        return self.fc(self.conv(x).mean([2, 3]))

class MockDataset:
    def __init__(self, data_root, split='train', transform=None):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        
    def __len__(self):
        return 100
        
    def __getitem__(self, idx):
        return {"data": torch.randn(3, 32, 32), "label": 0}

class MockTrainer:
    def __init__(self, max_epochs=10, accelerator='cpu'):
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        
    def fit(self, model, datamodule):
        return True

class SGDOptimizer:
    def __init__(self, params, lr=0.01, momentum=0, weight_decay=0):
        self.optimizer = SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

class StepLRScheduler:
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)


@pytest.fixture(scope="function")
def setup_registry():
    """Set up registries with mock components for testing"""
    # Register model component
    LIGHTNING_MODULE._module_dict["MockModel"] = MockModel
    
    # Register dataset component
    DATA._module_dict["MockDataset"] = MockDataset
    
    # Register trainer component
    TRAINER._module_dict["MockTrainer"] = MockTrainer
    
    # Register optimizer component
    OPTIMIZERS._module_dict["SGDOptimizer"] = SGDOptimizer
    
    # Register scheduler component
    SCHEDULERS._module_dict["StepLRScheduler"] = StepLRScheduler
    
    # Simple model for testing optimizers and schedulers
    model = MockModel()
    
    # Convert params to list to avoid generator exhaustion issues
    params = list(model.parameters())
    
    return {
        "model": model,
        "params": params
    }


def test_build_from_cfg(setup_registry):
    """Test build_from_cfg function"""
    model_cfg = {"type": "MockModel", "in_channels": 1, "num_classes": 5}
    model = build_from_cfg(model_cfg, LIGHTNING_MODULE)
    
    assert isinstance(model, MockModel)
    assert model.in_channels == 1
    assert model.num_classes == 5


def test_build_model(setup_registry):
    """Test build_model function"""
    model_cfg = {"type": "MockModel", "in_channels": 3, "num_classes": 10}
    model = build_model(model_cfg)
    
    assert isinstance(model, MockModel)
    assert model.in_channels == 3
    assert model.num_classes == 10


def test_build_dataset(setup_registry):
    """Test build_dataset function"""
    dataset_cfg = {"type": "MockDataset", "data_root": "/data", "split": "val"}
    dataset = build_dataset(dataset_cfg)
    
    assert isinstance(dataset, MockDataset)
    assert dataset.data_root == "/data"
    assert dataset.split == "val"


def test_build_trainer(setup_registry):
    """Test build_trainer function"""
    trainer_cfg = {"type": "MockTrainer", "max_epochs": 20, "accelerator": "gpu"}
    trainer = build_trainer(trainer_cfg)
    
    assert isinstance(trainer, MockTrainer)
    assert trainer.max_epochs == 20
    assert trainer.accelerator == "gpu"

def test_build_optimizer(setup_registry):
    """Test build_optimizer function"""
    params = setup_registry["params"]
    
    # Test direct dictionary config
    optimizer_cfg = {"type": "SGDOptimizer", "lr": 0.05, "momentum": 0.9}
    optimizer = build_optimizer(optimizer_cfg, params=params)
    
    assert isinstance(optimizer, SGD)
    assert optimizer.param_groups[0]["lr"] == 0.05
    assert optimizer.param_groups[0]["momentum"] == 0.9
    
    # Test with _instance key (simulating converted instance)
    optim_instance = SGD(params, lr=0.1)
    optimizer_cfg = {"_instance": optim_instance}
    optimizer = build_optimizer(optimizer_cfg)
    
    assert optimizer is optim_instance
    assert optimizer.param_groups[0]["lr"] == 0.1


def test_build_scheduler(setup_registry):
    """Test build_scheduler function"""
    params = setup_registry["params"]
    optimizer = SGD(params, lr=0.01)
    
    # Test direct dictionary config
    scheduler_cfg = {"type": "StepLRScheduler", "step_size": 5, "gamma": 0.5}
    scheduler = build_scheduler(scheduler_cfg, optimizer=optimizer)
    
    assert isinstance(scheduler, StepLR)
    assert scheduler.step_size == 5
    assert scheduler.gamma == 0.5
    
    # Test with _instance key (simulating converted instance)
    scheduler_instance = CosineAnnealingLR(optimizer, T_max=10)
    scheduler_cfg = {"_instance": scheduler_instance}
    scheduler = build_scheduler(scheduler_cfg, optimizer=optimizer)
    
    assert scheduler is scheduler_instance
    assert scheduler.T_max == 10

