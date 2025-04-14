import pytest
import torch
import torch.nn as nn
import lightning.pytorch as L
import os
import tempfile
from xinnovation.src.components.deploy.converter import ONNXConverter
from xinnovation.src.components.deploy.deploy import Deployer

class SimpleModel(L.LightningModule):
    """A simple model for testing deployment"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 16 * 16, 10)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

@pytest.fixture
def model():
    return SimpleModel()

@pytest.fixture
def input_sample():
    return torch.randn(1, 3, 32, 32)

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

def test_onnx_converter(model, input_sample, temp_dir):
    """Test ONNX converter functionality"""
    # Create converter
    converter = ONNXConverter()
    
    # Convert model
    save_path = os.path.join(temp_dir, 'model.onnx')
    output_path = converter.convert(model, input_sample, save_path)
    
    # Check if file exists
    assert os.path.exists(output_path)
    
    # Check if file is valid ONNX
    import onnx
    onnx_model = onnx.load(output_path)
    assert onnx_model is not None
    
    # Check input/output shapes
    assert len(onnx_model.graph.input) == 1
    assert len(onnx_model.graph.output) == 1
    
    # Check dynamic axes
    assert 'batch_size' in str(onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param)
    assert 'batch_size' in str(onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_param)
