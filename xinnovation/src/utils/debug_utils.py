from typing import Dict, Any, Union, List
import torch
import numpy as np
import torch.nn as nn


def check_nan_or_inf(tensor: Union[torch.Tensor, List[torch.Tensor], Dict[Any, torch.Tensor]],
                     active: bool=True,
                     name: str="tensor"):
    if not active:
        return
    
    # torch.nonzero(tensor.isnan()) => find the indices of the NaN values
    # torch.nonzero(tensor.isinf()) => find the indices of the Inf values
    
    if isinstance(tensor, torch.Tensor):
        if torch.isnan(tensor).any():
            raise ValueError(f"{name} contains NaN values")
        if torch.isinf(tensor).any():
            raise ValueError(f"{name} contains Inf values")
    elif isinstance(tensor, list):
        i = 0
        for t in tensor:
            check_nan_or_inf(t, f"{name}[{i}]")
            i += 1
    elif isinstance(tensor, dict):
        for k, v in tensor.items():
            check_nan_or_inf(v, f"{name}.{k}")
            