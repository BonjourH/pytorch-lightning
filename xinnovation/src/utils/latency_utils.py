import time
import torch
from typing import Callable

def measure_gpu_latency(func: Callable, *args, **kwargs) -> float:
    """
    测量 GPU 上函数的执行时间（确保同步 CUDA 流）。
    """
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # 同步并记录开始时间
    start_event.record()
    
    # 执行函数
    result = func(*args, **kwargs)
    
    # 同步并记录结束时间
    end_event.record()
    torch.cuda.synchronize()  # 确保所有操作完成
    
    # 返回执行时间（毫秒）
    latency_ms = start_event.elapsed_time(end_event)
    print(f"GPU Latency of {func.__name__}: {latency_ms:.6f} ms")
    return result

def measure_cpu_latency(func: Callable, *args, **kwargs) -> float:
    """
    测量 CPU 上函数的执行时间。
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    print(f"CPU Latency of {func.__name__}: {latency_ms:.6f} ms")
    return result


def measure_average_gpu_latency(func: Callable, *args, **kwargs) -> float:
    """
    测量 GPU 上函数的平均执行时间。
    """
    total_latency = 0
    num_runs = 100
    
    for _ in range(num_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        result = func(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        total_latency += start_event.elapsed_time(end_event)
        
    average_latency = total_latency / num_runs
    print(f"Average GPU Latency of {func.__name__}: {average_latency:.6f} ms")
    return result
