import os
import torch
import torch.nn as nn
from typing import Dict, Optional, Union, List
from pathlib import Path

from xinnovation.src.core.config import Config
from xinnovation.src.core.registry import MODEL, DEPLOY

def export(
    config: Union[str, Dict, Config],
    checkpoint: str,
    work_dir: str,
    export_format: str = 'onnx',
    gpu_ids: Optional[list] = None,
    seed: int = 42
):
    """导出流程
    
    Args:
        config (Union[str, Dict, Config]): 配置对象或配置文件路径
        checkpoint (str): 模型检查点路径
        work_dir (str): 工作目录
        export_format (str): 导出格式，支持 'onnx', 'torchscript', 'coreml'
        gpu_ids (list, optional): GPU ID列表
        seed (int): 随机种子
    """
    # 设置随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    # 加载配置
    if isinstance(config, (str, Path)):
        config = Config(filename=str(config))
    elif isinstance(config, dict):
        config = Config(cfg_dict=config)
        
    # 设置工作目录
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))
    device = torch.device(f'cuda:{gpu_ids[0]}' if gpu_ids else 'cpu')
    
    # 构建模型
    model = MODEL.build(config['model'])
    model = model.to(device)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 构建导出器
    if export_format == 'onnx':
        exporter = DEPLOY.build(
            config['deploy'].get('converter', dict(type='ONNXConverter')),
            model=model,
            work_dir=work_dir
        )
    elif export_format == 'torchscript':
        exporter = DEPLOY.build(
            config['deploy'].get('converter', dict(type='TorchScriptConverter')),
            model=model,
            work_dir=work_dir
        )
    elif export_format == 'coreml':
        exporter = DEPLOY.build(
            config['deploy'].get('converter', dict(type='CoreMLConverter')),
            model=model,
            work_dir=work_dir
        )
    else:
        raise ValueError(f'Unsupported export format: {export_format}')
        
    # 构建量化器（如果配置了）
    quantizer = None
    if 'quantizer' in config['deploy']:
        quantizer = DEPLOY.build(
            config['deploy']['quantizer'],
            model=model
        )
        
    # 构建剪枝器（如果配置了）
    pruner = None
    if 'pruner' in config['deploy']:
        pruner = DEPLOY.build(
            config['deploy']['pruner'],
            model=model
        )
        
    # 构建压缩器（如果配置了）
    compressor = None
    if 'compressor' in config['deploy']:
        compressor = DEPLOY.build(
            config['deploy']['compressor'],
            model=model
        )
        
    # 构建运行时优化器（如果配置了）
    runtime = None
    if 'runtime' in config['deploy']:
        runtime = DEPLOY.build(
            config['deploy']['runtime'],
            model=model
        )
        
    # 开始导出流程
    # 1. 应用量化（如果配置了）
    if quantizer is not None:
        model = quantizer.quantize(model)
        
    # 2. 应用剪枝（如果配置了）
    if pruner is not None:
        model = pruner.prune(model)
        
    # 3. 应用压缩（如果配置了）
    if compressor is not None:
        model = compressor.compress(model)
        
    # 4. 应用运行时优化（如果配置了）
    if runtime is not None:
        model = runtime.optimize(model)
        
    # 5. 导出模型
    exporter.convert(model) 