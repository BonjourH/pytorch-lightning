import torch
import pytest
import traceback
from xinnovation.src.components.lightning_module.detectors import FPNImageFeatureExtractor


def test_fpn_resnet18_img_feature_extractor():
    """测试FPNImageFeatureExtractor能够正确初始化"""
    backbone_config = {
        "type": "ImageFeatureExtractor",
        "backbone": "resnet18",
        "use_pretrained": False
    }
    
    neck_config = {
        "type": "FPN",
        "out_channels": 256
    }
    
    # 简单测试初始化不会抛出异常
    extractor = FPNImageFeatureExtractor(
        backbone=backbone_config,
        neck=neck_config,
        name="test_extractor"
    )
    
    x = torch.rand(1, 3, 224, 224)
    outputs = extractor(x)
    assert outputs is not None
    
    assert extractor.name == "test_extractor"
    assert hasattr(extractor, "backbone")
    assert hasattr(extractor, "neck")


def test_fpn_resnet18_img_feature_extractor_with_drop_scales():
    """测试FPNImageFeatureExtractor能够正确初始化"""
    backbone_config = {
        "type": "ImageFeatureExtractor",
        "backbone": "resnet18",
        "use_pretrained": False,
        "scales_to_drop": [4]
    }
    
    neck_config = {
        "type": "FPN",
        "out_channels": 256
    }
    
    # 简单测试初始化不会抛出异常
    extractor = FPNImageFeatureExtractor(
        backbone=backbone_config,
        neck=neck_config,
        name="test_extractor"
    )
    
    x = torch.rand(1, 3, 224, 224)
    features, scales = extractor(x)
    assert len(features) == 3
    assert len(scales) == 3
    assert scales == [8, 16, 32]
    assert features[0].shape == torch.Size([1, 256, 224 // 8, 224 // 8])
    assert features[1].shape == torch.Size([1, 256, 224 // 16, 224 // 16])
    assert features[2].shape == torch.Size([1, 256, 224 // 32, 224 // 32])
    
    assert extractor.name == "test_extractor"
    assert hasattr(extractor, "backbone")
    assert hasattr(extractor, "neck")
