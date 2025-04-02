from xinnovation.src.core import NORM_LAYERS
import torch.nn as nn

__all__ = ["LayerNorm", "GroupNorm", "BatchNorm", "InstanceNorm"]


@NORM_LAYERS.register_module()
class LayerNorm(nn.LayerNorm):
    """LayerNorm wrapper that handles configuration via registry.
    
    Args:
        normalized_shape (int or tuple): Input shape from an expected input of size
            [* x normalized_shape[0] x normalized_shape[1] x ... x normalized_shape[-1]]
        eps (float, optional): Value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine (bool, optional): If True, use learnable affine parameters. Default: True
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

@NORM_LAYERS.register_module()
class GroupNorm(nn.GroupNorm):
    """GroupNorm wrapper that handles configuration via registry.
    
    Args:
        num_groups (int): Number of groups to separate the channels into
        num_channels (int): Number of channels expected in input
        eps (float, optional): Value added to the denominator for numerical stability. Default: 1e-5
        affine (bool, optional): If True, use learnable affine parameters. Default: True
    """
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__(num_groups, num_channels, eps=eps, affine=affine)

@NORM_LAYERS.register_module()
class BatchNorm(nn.BatchNorm1d):
    """BatchNorm1d wrapper that handles configuration via registry.
    
    Args:
        num_features (int): C from an expected input of size (N, C, L)
        eps (float, optional): Value added to the denominator for numerical stability. Default: 1e-5
        momentum (float, optional): Value used for the running_mean and running_var computation. Default: 0.1
        affine (bool, optional): If True, use learnable affine parameters. Default: True
        track_running_stats (bool, optional): If True, track running mean and variance. Default: True
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

@NORM_LAYERS.register_module()
class InstanceNorm(nn.InstanceNorm1d):
    """InstanceNorm1d wrapper that handles configuration via registry.
    
    Args:
        num_features (int): C from an expected input of size (N, C, L)
        eps (float, optional): Value added to the denominator for numerical stability. Default: 1e-5
        momentum (float, optional): Value used for the running_mean and running_var computation. Default: 0.1
        affine (bool, optional): If True, use learnable affine parameters. Default: True
        track_running_stats (bool, optional): If True, track running mean and variance. Default: False
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

