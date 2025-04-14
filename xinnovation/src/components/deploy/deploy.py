from xinnovation.src.core.registry import Registry, DEPLOY, CONVERTERS, QUANTIZERS, PRUNERS
import lightning.pytorch as L
from typing import Dict, List, Optional, Any, Union, Tuple
import torch
from .converter import ONNXConverter


@DEPLOY.register_module()
class Deployer:
    """Lightning model deployment manager"""
    
    def __init__(self, onnx_converter_cfg: Dict, quantizer_cfg: Dict=None, pruner_cfg: Dict=None):
        self.onnx_converter = CONVERTERS.build(onnx_converter_cfg)
        if quantizer_cfg is not None:
            self.quantizer = QUANTIZERS.build(quantizer_cfg)
        if pruner_cfg is not None:
            self.pruner = PRUNERS.build(pruner_cfg)
            
    def convert_to_onnx(self, model: L.LightningModule, input_sample: Union[torch.Tensor, Tuple], save_path: str) -> str:
        return self.onnx_converter.convert(model, input_sample, save_path)
