import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from collections import defaultdict
import time
import psutil
import os
import gc
from torch.autograd import Variable
import logging

class MemoryTracker:
    """Track memory usage of PyTorch models and tensors."""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset()
    
    def reset(self):
        """Reset memory tracking."""
        self.tensor_sizes = defaultdict(list)
        self.max_memory = 0
        self.current_memory = 0
    
    def track_tensor(self, tensor: torch.Tensor, name: str):
        """Track memory usage of a tensor."""
        if tensor.device == self.device:
            size = tensor.element_size() * tensor.nelement()
            self.tensor_sizes[name].append(size)
            self.current_memory += size
            self.max_memory = max(self.max_memory, self.current_memory)
    
    def report(self) -> Dict[str, float]:
        """Get memory usage report."""
        report = {
            'max_memory_mb': self.max_memory / (1024 * 1024),
            'current_memory_mb': self.current_memory / (1024 * 1024)
        }
        
        for name, sizes in self.tensor_sizes.items():
            report[f'{name}_mb'] = sum(sizes) / (1024 * 1024)
        
        if torch.cuda.is_available():
            report['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
            report['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
        
        report['cpu_memory_percent'] = psutil.Process(os.getpid()).memory_percent()
        
        return report

class ModelAnalyzer:
    """Analyze PyTorch model structure and parameters."""
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def summarize(self) -> Dict[str, Union[int, float, str]]:
        """Get model summary statistics."""
        summary = {
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'layer_count': len(list(self.model.modules())),
            'parameter_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        }
        
        # Count parameters by type
        type_counts = defaultdict(int)
        for name, module in self.model.named_modules():
            type_counts[module.__class__.__name__] += 1
        summary['layer_types'] = dict(type_counts)
        
        return summary
    
    def analyze_activations(self,
                          sample_input: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Analyze layer activations."""
        activations = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = {
                    'mean': float(output.mean().item()),
                    'std': float(output.std().item()),
                    'min': float(output.min().item()),
                    'max': float(output.max().item()),
                    'shape': list(output.shape)
                }
            return hook
        
        # Register hooks
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(sample_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return activations
    
    def analyze_gradients(self,
                        sample_input: Dict[str, torch.Tensor],
                        sample_target: Dict[str, torch.Tensor],
                        criterion: nn.Module) -> Dict[str, np.ndarray]:
        """Analyze parameter gradients."""
        gradients = {}
        
        # Forward and backward pass
        self.model.train()
        outputs = self.model(sample_input)
        loss = criterion(outputs, sample_target)
        loss.backward()
        
        # Collect gradients
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = {
                    'mean': float(param.grad.mean().item()),
                    'std': float(param.grad.std().item()),
                    'min': float(param.grad.min().item()),
                    'max': float(param.grad.max().item()),
                    'norm': float(param.grad.norm().item())
                }
        
        # Reset gradients
        self.model.zero_grad()
        
        return gradients
    
    def find_unused_parameters(self) -> List[str]:
        """Find parameters that don't receive gradients."""
        unused = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                unused.append(name)
            elif hasattr(param, 'grad') and param.grad is None:
                unused.append(name)
        
        return unused

class PerformanceProfiler:
    """Profile model performance and bottlenecks."""
    
    def __init__(self):
        self.timings = defaultdict(list)
    
    def profile_forward(self,
                       model: nn.Module,
                       sample_input: Dict[str, torch.Tensor],
                       num_runs: int = 100) -> Dict[str, float]:
        """Profile forward pass performance."""
        timings = defaultdict(list)
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.time()
                timings[name].append(end - hook.start_time)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                hook.start_time = end
            return hook
        
        # Register hooks
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.MultiheadAttention)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Warmup
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        # Profile runs
        total_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.time()
                
                # Set start time for first layer
                for hook in hooks:
                    hook.start_time = start
                
                _ = model(sample_input)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.time()
                total_times.append(end - start)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute statistics
        results = {
            'total_time': np.mean(total_times),
            'total_std': np.std(total_times)
        }
        
        for name, times in timings.items():
            results[f'{name}_time'] = np.mean(times)
            results[f'{name}_std'] = np.std(times)
            results[f'{name}_percent'] = 100 * np.mean(times) / results['total_time']
        
        return results

class GradientChecker:
    """Check gradient computation correctness."""
    
    @staticmethod
    def check_gradients(model: nn.Module,
                       criterion: nn.Module,
                       sample_input: Dict[str, torch.Tensor],
                       sample_target: Dict[str, torch.Tensor],
                       eps: float = 1e-3) -> Dict[str, bool]:
        """Check gradients using finite differences."""
        results = {}
        
        # Get analytical gradients
        model.train()
        outputs = model(sample_input)
        loss = criterion(outputs, sample_target)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            
            analytical_grad = param.grad.clone()
            param.grad.zero_()
            
            # Compute numerical gradients
            numerical_grad = torch.zeros_like(param)
            for i in range(param.numel()):
                # Flatten parameter
                orig = param.view(-1)[i].item()
                
                # Compute f(x + h)
                param.view(-1)[i] = orig + eps
                outputs = model(sample_input)
                loss_plus = criterion(outputs, sample_target)
                
                # Compute f(x - h)
                param.view(-1)[i] = orig - eps
                outputs = model(sample_input)
                loss_minus = criterion(outputs, sample_target)
                
                # Restore original value
                param.view(-1)[i] = orig
                
                # Compute numerical gradient
                numerical_grad.view(-1)[i] = (loss_plus - loss_minus) / (2 * eps)
            
            # Compare gradients
            rel_error = (analytical_grad - numerical_grad).abs() / (analytical_grad.abs() + numerical_grad.abs() + 1e-10)
            max_error = rel_error.max().item()
            
            results[name] = {
                'max_error': max_error,
                'passed': max_error < 1e-5
            }
        
        return results

def setup_debug_logging(log_file: str = 'debug.log'):
    """Setup detailed logging for debugging."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Add debug logging for PyTorch
    logging.getLogger('torch').setLevel(logging.DEBUG)
    logging.getLogger('torch.utils.data').setLevel(logging.DEBUG)
    
    return logging.getLogger('debug') 