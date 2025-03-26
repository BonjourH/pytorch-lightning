import os
import torch
from typing import Dict, Optional, Union, List
from pathlib import Path
import json

from xinnovation.src.core.config import Config
from xinnovation.src.core.registry import MODEL, DATA, EVALUATION

def test(
    config: Union[str, Dict, Config],
    checkpoint: str,
    work_dir: str,
    gpu_ids: Optional[list] = None,
    show: bool = False,
    show_dir: Optional[str] = None,
    seed: int = 42
):
    """测试流程
    
    Args:
        config (Union[str, Dict, Config]): 配置对象或配置文件路径
        checkpoint (str): 模型检查点路径
        work_dir (str): 工作目录
        gpu_ids (list, optional): GPU ID列表
        show (bool): 是否显示结果
        show_dir (str, optional): 结果保存目录
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
    
    # 构建测试数据集
    test_dataset = DATA.build(config['data']['test'])
    
    # 构建数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['test']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 构建评估指标
    metrics = []
    for metric_cfg in config['evaluation']['metrics']:
        metrics.append(EVALUATION.build(metric_cfg))
        
    # 构建结果分析器
    analyzers = []
    for analyzer_cfg in config['evaluation']['analyzers']:
        analyzers.append(EVALUATION.build(analyzer_cfg))
        
    # 构建可视化器
    visualizers = []
    for visualizer_cfg in config['evaluation']['visualizers']:
        visualizers.append(EVALUATION.build(visualizer_cfg))
        
    # 设置结果保存目录
    if show and show_dir is None:
        show_dir = work_dir / 'results'
    if show_dir:
        show_dir = Path(show_dir)
        show_dir.mkdir(parents=True, exist_ok=True)
        
    # 开始测试
    results = []
    with torch.no_grad():
        for batch in test_loader:
            # 将数据移到设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 前向传播
            outputs = model(batch)
            
            # 收集结果
            results.append(outputs)
            
            # 可视化结果
            if show and show_dir:
                for visualizer in visualizers:
                    visualizer.visualize(
                        batch,
                        outputs,
                        save_dir=show_dir
                    )
                    
    # 计算评估指标
    for metric in metrics:
        metric.update(results)
    metrics_results = {metric.__class__.__name__: metric.compute() 
                      for metric in metrics}
    
    # 分析结果
    for analyzer in analyzers:
        analyzer.analyze(results)
    analysis_results = {analyzer.__class__.__name__: analyzer.get_results() 
                       for analyzer in analyzers}
    
    # 保存结果
    results_file = work_dir / 'test_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'metrics': metrics_results,
            'analysis': analysis_results
        }, f, indent=2)
        
    return metrics_results, analysis_results 