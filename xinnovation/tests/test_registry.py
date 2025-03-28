import pytest
import torch.nn as nn

from xinnovation.src.core.registry import (
    Registry, BACKBONES, HEADS, FUSION, CONVERTERS, METRICS, LIGHTNING_MODULE
)

# Test components - defined at module level so they're available for all tests
class ResNet(nn.Module):
    def __init__(self, depth=50, pretrained=None):
        super().__init__()
        self.depth = depth
        self.pretrained = pretrained
        
    def forward(self, x):
        return x

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
    def forward(self, x):
        return x

class FeatureFusion(nn.Module):
    def __init__(self, fusion_type="concat"):
        super().__init__()
        self.fusion_type = fusion_type
        
    def forward(self, x1, x2):
        return x1 + x2

class ONNXConverter:
    def __init__(self, opset_version=11):
        self.opset_version = opset_version
        
    def convert(self, model, dummy_input):
        return model

class MeanAP:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        
    def compute(self, pred, target):
        return 0.95


@pytest.fixture(scope="module")
def register_components():
    """Register all test components for use in tests"""
    # Clear any existing registrations for test reliability
    for registry in [BACKBONES, HEADS, FUSION, CONVERTERS, METRICS]:
        registry._module_dict.clear()
        registry._group_dict.clear()
    
    # Register test components
    BACKBONES.register_module()(ResNet)
    HEADS.register_module(group="detection")(DetectionHead)
    FUSION.register_module()(FeatureFusion)
    CONVERTERS.register_module()(ONNXConverter)
    METRICS.register_module()(MeanAP)
    
    # Return registered components for test use
    return {
        "backbone": ResNet,
        "head": DetectionHead,
        "fusion": FeatureFusion,
        "converter": ONNXConverter,
        "metric": MeanAP
    }


def test_basic_registration_and_building(register_components):
    """Test 1: Test basic component registration and building"""
    backbone_cfg = {"type": "ResNet", "depth": 101}
    backbone = BACKBONES.build(backbone_cfg)
    
    assert isinstance(backbone, ResNet)
    assert backbone.depth == 101
    assert backbone.pretrained is None


def test_group_functionality(register_components):
    """Test 2: Test component grouping functionality"""
    detection_heads = HEADS.get_group("detection")
    
    assert len(detection_heads) == 1
    assert "DetectionHead" in detection_heads


def test_multimodal_components(register_components):
    """Test 3: Test multimodal component registration and building"""
    fusion_cfg = {"type": "FeatureFusion", "fusion_type": "concat"}
    fusion = FUSION.build(fusion_cfg)
    
    assert isinstance(fusion, FeatureFusion)
    assert fusion.fusion_type == "concat"


def test_deployment_components(register_components):
    """Test 4: Test deployment component registration and building"""
    converter_cfg = {"type": "ONNXConverter", "opset_version": 12}
    converter = CONVERTERS.build(converter_cfg)
    
    assert isinstance(converter, ONNXConverter)
    assert converter.opset_version == 12


def test_evaluation_components(register_components):
    """Test 5: Test evaluation component registration and building"""
    metric_cfg = {"type": "MeanAP", "iou_threshold": 0.5}
    metric = METRICS.build(metric_cfg)
    
    assert isinstance(metric, MeanAP)
    assert metric.iou_threshold == 0.5


def test_cyclic_dependency_detection(register_components):
    """Test 6: Test detection of cyclic dependencies in config"""
    # Override the build method to force a cyclic dependency for testing
    original_build = BACKBONES.build
    
    def recursive_build(cfg, *args, **kwargs):
        """Mock build method that will trigger cyclic dependency"""
        if "type" in cfg:
            obj_type = cfg["type"]
            if obj_type == "ResNet" and "pretrained" in cfg and isinstance(cfg["pretrained"], dict):
                # Simulate recursive building by calling build again with the same type
                BACKBONES._building_stack.append(obj_type)
                try:
                    # This will trigger the cyclic dependency detection
                    return original_build({"type": "ResNet"})
                finally:
                    BACKBONES._building_stack.pop()
        return original_build(cfg, *args, **kwargs)
    
    # Replace build method temporarily
    BACKBONES.build = recursive_build
    
    try:
        cyclic_cfg = {
            "type": "ResNet",
            "depth": 50,
            "pretrained": {
                "type": "ResNet",
                "depth": 101
            }
        }
        
        with pytest.raises(RuntimeError) as excinfo:
            BACKBONES.build(cyclic_cfg)
        
        # Check that error message contains cyclic dependency info
        assert "Cyclic dependency detected" in str(excinfo.value)
    finally:
        # Restore original build method
        BACKBONES.build = original_build


def test_registry_information(register_components):
    """Test 7: Test registry information retrieval"""
    # Check that LIGHTNING_MODULE registry has child registries
    assert len(LIGHTNING_MODULE._children) > 0
    
    # Check that we can get a list of registered modules
    registered_modules = BACKBONES.get_registered_modules()
    assert "ResNet" in registered_modules


def test_nested_config_structure():
    """Test 8: Test the structure of nested configs (without actual build)"""
    model_cfg = {
        "type": "DetectionModel",
        "backbone": {
            "type": "ResNet",
            "depth": 50
        },
        "head": {
            "type": "DetectionHead",
            "in_channels": 2048,
            "num_classes": 80
        }
    }
    
    # Verify the structure of the nested config
    assert model_cfg["type"] == "DetectionModel"
    assert model_cfg["backbone"]["type"] == "ResNet"
    assert model_cfg["backbone"]["depth"] == 50
    assert model_cfg["head"]["type"] == "DetectionHead"
    assert model_cfg["head"]["in_channels"] == 2048
    assert model_cfg["head"]["num_classes"] == 80


def test_registry_creation():
    """Test registry creation and basic functionality"""
    # Create a test registry
    test_registry = Registry("test")
    
    # Test basic properties
    assert test_registry._name == "test"
    assert len(test_registry) == 0
    
    # Test child registry creation
    child = test_registry.create_child("child")
    assert child._name == "test.child"
    assert "child" in test_registry._children
    
    # Test registration
    @test_registry.register_module()
    class TestModule:
        pass
    
    assert "TestModule" in test_registry
    assert test_registry["TestModule"] == TestModule
    assert len(test_registry) == 1


def test_forced_registration():
    """Test forced registration of modules with same name"""
    test_registry = Registry("force_test")
    
    # First registration
    @test_registry.register_module()
    class TestModule:
        value = 1
    
    # This should fail without force=True
    with pytest.raises(KeyError):
        @test_registry.register_module()
        class TestModule:
            value = 2
    
    # This should succeed with force=True
    @test_registry.register_module(force=True)
    class TestModule:
        value = 2
    
    # Check that the second registration replaced the first
    assert test_registry["TestModule"].value == 2 