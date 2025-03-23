# registry.py

from typing import Dict, Any, Callable, Optional, Type, Union, List
import inspect
from collections import defaultdict

class Registry:
    """
    A registry for managing modular components in a Lightning project.
    
    This registry system is similar to the one used in mmdet3D but adapted
    for PyTorch Lightning applications. It allows for registration and building
    of models, transforms, datasets, etc. based on configuration dictionaries.
    """
    
    def __init__(self, name: str):
        """
        Initialize the registry.
        
        Args:
            name (str): Registry name
        """
        self._name = name
        self._module_dict: Dict[str, Type] = {}
        self._children: Dict[str, 'Registry'] = {}
        self._group_dict: Dict[str, List[str]] = defaultdict(list)
    
    def __repr__(self):
        return f"Registry(name={self._name}, items={list(self._module_dict.keys())})"
    
    def __len__(self):
        return len(self._module_dict)
    
    def __contains__(self, key):
        return key in self._module_dict
    
    def __getitem__(self, key):
        return self.get(key)
    
    def create_child(self, name: str) -> 'Registry':
        """
        Create a child registry.
        
        Args:
            name (str): Name of child registry
            
        Returns:
            Registry: The child registry
        """
        if name in self._children:
            return self._children[name]
            
        child = Registry(f"{self._name}.{name}")
        self._children[name] = child
        return child
    
    def register_module(
        self, 
        name: Optional[str] = None, 
        module: Optional[Type] = None, 
        force: bool = False,
        group: Optional[str] = None
    ) -> Callable:
        """
        Register a module.
        
        Args:
            name (str, optional): Module name to be registered. If not specified, 
                                the class name will be used.
            module (Type, optional): Module class/function to be registered. 
                                If not specified, this method can be used as a decorator.
            force (bool): Whether to override an existing module with the same name.
                        Default: False.
            group (str, optional): Group to categorize this module. Useful for organizing
                                modules by functionality or type.
        
        Returns:
            Callable: The registered module or a decorator function.
        """
        
        # Used as a decorator: @MODELS.register_module()
        if module is None:
            def _register(cls):
                cls_name = name
                if cls_name is None:
                    cls_name = cls.__name__
                self._register_module(cls_name, cls, force, group)
                return cls
                
            return _register
        
        # Used as a function: MODELS.register_module("MyModel", MyModel)
        else:
            module_name = name
            if module_name is None:
                module_name = module.__name__
            self._register_module(module_name, module, force, group)
            return module
    
    def _register_module(self, name: str, module: Type, force: bool = False, group: Optional[str] = None) -> None:
        """
        Internal implementation for registering a module.
        
        Args:
            name (str): Module name to be registered.
            module (Type): Module class/function to be registered.
            force (bool): Whether to override an existing module with the same name.
            group (str, optional): Group name for categorizing modules.
        """
        if not force and name in self._module_dict:
            raise KeyError(f"{name} is already registered in {self._name}")
        
        self._module_dict[name] = module
        
        if group is not None:
            self._group_dict[group].append(name)
    
    def get(self, key: str) -> Type:
        """
        Get the module from the registry.
        
        Args:
            key (str): The name of the module to retrieve.
        
        Returns:
            Type: The registered module.
        """
        if key not in self._module_dict:
            raise KeyError(f"{key} is not registered in {self._name}. "
                          f"Available modules are: {list(self._module_dict.keys())}")
        
        return self._module_dict[key]
    
    def get_group(self, group_name: str) -> List[str]:
        """
        Get names of all modules in a specific group.
        
        Args:
            group_name (str): Name of the group
            
        Returns:
            List[str]: List of module names in the group
        """
        return self._group_dict[group_name]
    
    def build(self, cfg: Union[Dict[str, Any], Type], *args, **kwargs) -> Any:
        """
        Build a module from config dict.
        
        Args:
            cfg (Union[Dict[str, Any], Type]): Config dict or actual module class.
                                            If dict, it should contain the key "type".
            *args: Additional arguments for the module constructor.
            **kwargs: Additional keyword arguments for the module constructor.
        
        Returns:
            Any: The constructed module.
        """
        if cfg is None:
            return None
            
        if not isinstance(cfg, dict):
            # If cfg is already a type/class, return an instance
            if inspect.isclass(cfg):
                return cfg(*args, **kwargs)
            # If it's already an instance, return it
            return cfg
            
        cfg = cfg.copy()
        
        if "type" not in cfg:
            raise KeyError("cfg must contain the key 'type'")
        
        obj_type = cfg.pop("type")
        obj_cls = self.get(obj_type)
        
        try:
            # Handle nested configs
            for key, val in cfg.items():
                if isinstance(val, dict) and "type" in val and key in kwargs:
                    cfg[key] = self.build(val)
                    
            return obj_cls(*args, **kwargs, **cfg)
        except Exception as e:
            # Show more informative error message
            params = inspect.signature(obj_cls.__init__).parameters
            param_str = ", ".join([f"{k}={v.default}" if v.default is not inspect.Parameter.empty else k 
                                 for k, v in params.items() if k != 'self'])
            
            raise TypeError(f"Failed to initialize {obj_type}: {e}\n"
                           f"Expected parameters: {param_str}")

    def get_module_dict(self) -> Dict[str, Type]:
        """Get the module dict."""
        return self._module_dict.copy()
        
    def get_registered_modules(self) -> List[str]:
        """Get list of registered module names."""
        return list(self._module_dict.keys())


# Create registries for different components (similar to mmdet3D)
MODELS = Registry('models')
BACKBONES = MODELS.create_child('backbone')
ATTENTION = MODELS.create_child('attention')
NORM_LAYERS = MODELS.create_child('norm_layer')
FEEDFORWARD_NETWORK = MODELS.create_child('feedforward_network')
HEADS = MODELS.create_child('head')
NECKS = MODELS.create_child('neck')
PLUGIN_LAYERS = MODELS.create_child('plugin_layer')
POSITIONAL_ENCODING = MODELS.create_child('positional_encoding')


DATASETS = Registry('dataset')
TRANSFORMS = Registry('transform')
CALLBACKS = Registry('callback')
LOSSES = Registry('loss')
OPTIMIZERS = Registry('optimizer')
LR_SCHEDULERS = Registry('lr_scheduler')
METRICS = Registry('metric')


# Usage example
if __name__ == "__main__":
    import torch.nn as nn
    from lightning.pytorch import LightningModule
    
    # Register a backbone model
    @BACKBONES.register_module()
    class ResNet(nn.Module):
        def __init__(self, depth=50, pretrained=None):
            super().__init__()
            self.depth = depth
            self.pretrained = pretrained
            
        def forward(self, x):
            return x
    
    # Register a model with a specific name and group
    @MODELS.register_module(name="MyNetwork", group="segmentation")
    class SegmentationModel(LightningModule):
        def __init__(self, backbone, learning_rate=0.001):
            super().__init__()
            self.save_hyperparameters()
            
            if isinstance(backbone, dict):
                self.backbone = BACKBONES.build(backbone)
            else:
                self.backbone = backbone
                
        def forward(self, x):
            return self.backbone(x)
            
        def training_step(self, batch, batch_idx):
            # Training implementation
            pass
    
    # Build from config
    model_cfg = {
        "type": "MyNetwork",
        "backbone": {
            "type": "ResNet",
            "depth": 101,
            "pretrained": "path/to/weights.pth"
        },
        "learning_rate": 0.0005
    }
    
    model = MODELS.build(model_cfg)
    print(f"Built model: {model}")
    print(f"Backbone: {model.backbone}")
    
    # Get all models in the segmentation group
    print(f"Segmentation models: {MODELS.get_group('segmentation')}")