from xinnovation.src.core import DROPOUT
import torch.nn as nn

__all__ = ["Dropout"]

@DROPOUT.register_module()
class Dropout(nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__(p=p, inplace=inplace)
