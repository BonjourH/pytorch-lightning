from xinnovation.src.core import ACTIVATION
import torch
import torch.nn as nn

__all__ = ["ReLU", "GELU", "SiLU", "Mish"]

@ACTIVATION.register_module()
class ReLU(nn.ReLU):
    def __init__(self, inplace: bool = False):
        super().__init__(inplace=inplace)


@ACTIVATION.register_module()
class GELU(nn.GELU):
    def __init__(self, inplace: bool = False):
        super().__init__(inplace=inplace)


@ACTIVATION.register_module()
class SiLU(nn.SiLU):
    def __init__(self, inplace: bool = False):
        super().__init__(inplace=inplace)


@ACTIVATION.register_module()
class Mish(nn.Mish):
    def __init__(self, inplace: bool = False):
        super().__init__(inplace=inplace)








