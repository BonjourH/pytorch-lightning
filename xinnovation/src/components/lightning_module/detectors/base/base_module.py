import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Any, List, Tuple
from abc import ABCMeta, abstractmethod
import torchviz
from torchinfo import summary
from pathlib import Path

__all__ = ["BaseModule"]

class BaseModule(nn.Module, metaclass=ABCMeta):
    """Base class for detectors."""
    def __init__(self, init_cfg: Union[dict, List[dict], None] = None):
        super(BaseModule, self).__init__()
        # NOTE init_cfg can be defined in different levels, but init_cfg
        # in low levels has a higher priority.

        super().__init__()
        # define default value of init_cfg instead of hard code
        # in init_weights() function
        self._is_init = False

        self.init_cfg = copy.deepcopy(init_cfg)

    @property
    def is_init(self):
        return self._is_init

    @is_init.setter
    def is_init(self, value):
        self._is_init = value
        
    def init_weights(self):
        """Initialize the weights."""

        is_top_level_module = False
        # check if it is top-level module
        if not hasattr(self, '_params_init_info'):
            # The `_params_init_info` is used to record the initialization
            # information of the parameters
            # the key should be the obj:`nn.Parameter` of model and the value
            # should be a dict containing
            # - init_info (str): The string that describes the initialization.
            # - tmp_mean_value (FloatTensor): The mean of the parameter,
            #       which indicates whether the parameter has been modified.
            # this attribute would be deleted after all parameters
            # is initialized.
            self._params_init_info = defaultdict(dict)
            is_top_level_module = True

            # Initialize the `_params_init_info`,
            # When detecting the `tmp_mean_value` of
            # the corresponding parameter is changed, update related
            # initialization information
            for name, param in self.named_parameters():
                self._params_init_info[param][
                    'init_info'] = f'The value is the same before and ' \
                                   f'after calling `init_weights` ' \
                                   f'of {self.__class__.__name__} '
                self._params_init_info[param][
                    'tmp_mean_value'] = param.data.mean().cpu()
                    # pass `params_init_info` to all submodules
            # All submodules share the same `params_init_info`,
            # so it will be updated when parameters are
            # modified at any level of the model.
            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info

        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(
                    f'initialize {module_name} with init_cfg {self.init_cfg}',
                    logger='current',
                    level=logging.DEBUG)

                init_cfgs = self.init_cfg
                if isinstance(self.init_cfg, dict):
                    init_cfgs = [self.init_cfg]

                # PretrainedInit has higher priority than any other init_cfg.
                # Therefore we initialize `pretrained_cfg` last to overwrite
                # the previous initialized weights.
                # See details in https://github.com/open-mmlab/mmengine/issues/691 # noqa E501
                other_cfgs = []
                pretrained_cfg = []
                for init_cfg in init_cfgs:
                    assert isinstance(init_cfg, dict)
                    if (init_cfg['type'] == 'Pretrained'
                            or init_cfg['type'] is PretrainedInit):
                        pretrained_cfg.append(init_cfg)
                    else:
                        other_cfgs.append(init_cfg)
                                       initialize(self, other_cfgs)

            for m in self.children():
                if is_model_wrapper(m) and not hasattr(m, 'init_weights'):
                    m = m.module
                if hasattr(m, 'init_weights') and not getattr(
                        m, 'is_init', False):
                    m.init_weights()
                    # users may overload the `init_weights`
                    update_init_info(
                        m,
                        init_info=f'Initialized by '
                        f'user-defined `init_weights`'
                        f' in {m.__class__.__name__} ')
            if self.init_cfg and pretrained_cfg:
                initialize(self, pretrained_cfg)
            self._is_init = True
        else:
            print_log(
                f'init_weights of {self.__class__.__name__} has '
                f'been called more than once.',
                logger='current',
                level=logging.WARNING)

        if is_top_level_module:
            self._dump_init_info()

            for sub_module in self.modules():
                del sub_module._params_init_info
    
    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f'\ninit_cfg={self.init_cfg}'
        return s
