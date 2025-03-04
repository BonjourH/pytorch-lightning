import torch
import time
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
import pandas as pd
import matplotlib.pyplot as plt
import wandb

class PerformanceProfiler:
    """Performance profiling tools."""
    
    def __init__(self):
        self.timing_stats = defaultdict(list)
        self.memory_stats = defaultdict(list)
        self.flops_stats = defaultdict(list)
        
    def profile_model(self, 
                     model: torch.nn.Module,
                     sample_input: Dict,
                     num_runs: int = 100) -> Dict[str, float]:
        """Profile model performance.
        
        Args:
            model: Model to profile
            sample_input: Sample input for the model
            num_runs: Number of runs for averaging
            
        Returns:
            Dict containing performance metrics
        """
        model.eval()
        metrics = {}
        
        # Time performance
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                model(sample_input)
            
            # Measure time
            start_time = time.time()
            for _ in range(num_runs):
                model(sample_input)
            avg_time = (time.time() - start_time) / num_runs
            metrics['inference_time'] = avg_time * 1000  # ms
        
        # Profile memory
        with torch.no_grad():
            torch.cuda.reset_peak_memory_stats()
            model(sample_input)
            max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            metrics['peak_memory'] = max_memory
        
        # Profile detailed operations
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True
        ) as prof:
            with record_function("model_inference"):
                model(sample_input)
        
        # Process profiler results
        events = prof.key_averages()
        total_flops = 0
        operation_times = defaultdict(float)
        
        for evt in events:
            if evt.flops > 0:
                total_flops += evt.flops
            operation_times[evt.key] = evt.cpu_time_total
        
        metrics['total_flops'] = total_flops
        metrics['operation_times'] = dict(operation_times)
        
        return metrics
    
    def profile_batch(self,
                     name: str,
                     start_time: float,
                     end_time: float,
                     memory_used: float):
        """Record batch processing statistics.
        
        Args:
            name: Name of the batch/operation
            start_time: Start timestamp
            end_time: End timestamp
            memory_used: Memory usage in bytes
        """
        duration = end_time - start_time
        self.timing_stats[name].append(duration)
        self.memory_stats[name].append(memory_used)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of profiling statistics.
        
        Returns:
            Dict containing timing and memory statistics
        """
        summary = {
            'timing': {},
            'memory': {}
        }
        
        # Process timing stats
        for name, times in self.timing_stats.items():
            summary['timing'][f'{name}_mean'] = np.mean(times)
            summary['timing'][f'{name}_std'] = np.std(times)
            summary['timing'][f'{name}_min'] = np.min(times)
            summary['timing'][f'{name}_max'] = np.max(times)
        
        # Process memory stats
        for name, mems in self.memory_stats.items():
            summary['memory'][f'{name}_mean'] = np.mean(mems) / 1024 / 1024  # MB
            summary['memory'][f'{name}_peak'] = np.max(mems) / 1024 / 1024   # MB
        
        return summary
    
    def visualize_performance(self) -> Dict[str, wandb.Object]:
        """Create visualizations of performance metrics.
        
        Returns:
            Dict of wandb visualization objects
        """
        vis_dict = {}
        
        # Create timing distribution plot
        fig, ax = plt.subplots(figsize=(10, 6))
        data = []
        labels = []
        
        for name, times in self.timing_stats.items():
            data.append(times)
            labels.append(name)
        
        ax.boxplot(data, labels=labels)
        ax.set_ylabel('Time (s)')
        ax.set_title('Operation Timing Distribution')
        plt.xticks(rotation=45)
        
        vis_dict['timing_distribution'] = wandb.Image(
            self._fig2img(fig)
        )
        plt.close(fig)
        
        # Create memory usage plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for name, mems in self.memory_stats.items():
            mems_mb = np.array(mems) / 1024 / 1024  # Convert to MB
            ax.plot(mems_mb, label=name)
        
        ax.set_xlabel('Batch')
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Memory Usage Over Time')
        ax.legend()
        
        vis_dict['memory_usage'] = wandb.Image(
            self._fig2img(fig)
        )
        plt.close(fig)
        
        return vis_dict
    
    def export_stats(self, path: str):
        """Export statistics to CSV file."""
        # Create DataFrame for timing stats
        timing_df = pd.DataFrame(self.timing_stats)
        timing_df.to_csv(f"{path}_timing.csv", index=False)
        
        # Create DataFrame for memory stats
        memory_df = pd.DataFrame(self.memory_stats)
        memory_df.to_csv(f"{path}_memory.csv", index=False)
    
    @staticmethod
    def _fig2img(fig: plt.Figure) -> np.ndarray:
        """Convert matplotlib figure to numpy array."""
        import io
        from PIL import Image
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        return np.array(img)

class BatchProfiler:
    """Context manager for profiling batch operations."""
    
    def __init__(self, profiler: PerformanceProfiler, name: str):
        self.profiler = profiler
        self.name = name
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = torch.cuda.memory_allocated()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated()
            memory_used = end_memory - self.start_memory
            
            self.profiler.profile_batch(
                self.name,
                self.start_time,
                end_time,
                memory_used
            ) 