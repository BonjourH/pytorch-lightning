import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from xinnovation.src.core.registry import STRATEGIES

@STRATEGIES.register_module()
class DDPStrategy:
    """Distributed Data Parallel training strategy.
    
    Args:
        model: PyTorch model
        device_ids (list): List of device IDs
        output_device (int): Output device ID
        find_unused_parameters (bool): Whether to find unused parameters
    """
    
    def __init__(
        self,
        model: nn.Module,
        device_ids=None,
        output_device=None,
        find_unused_parameters=False
    ):
        self.model = model
        self.device_ids = device_ids
        self.output_device = output_device
        self.find_unused_parameters = find_unused_parameters
        
        # Initialize distributed environment
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')
            
        # Wrap model with DDP
        self.ddp_model = DDP(
            model,
            device_ids=device_ids,
            output_device=output_device,
            find_unused_parameters=find_unused_parameters
        )
        
    def forward(self, *args, **kwargs):
        """Forward pass through DDP model."""
        return self.ddp_model(*args, **kwargs)
        
    def train(self):
        """Set model to training mode."""
        self.ddp_model.train()
        
    def eval(self):
        """Set model to evaluation mode."""
        self.ddp_model.eval()
        
    def state_dict(self):
        """Get model state dict."""
        return self.ddp_model.module.state_dict()
        
    def load_state_dict(self, state_dict):
        """Load model state dict."""
        return self.ddp_model.module.load_state_dict(state_dict)
        
    def __getattr__(self, name):
        """Get attribute from DDP model."""
        return getattr(self.ddp_model, name) 