import os
import torch
import torch.nn as nn
from typing import Dict, Optional, Union
from pathlib import Path

from xinnovation.src.core.config import Config
from xinnovation.src.core.registry import MODEL, DATA, TRAINER, EVALUATION
from xinnovation.src.components.trainer.strategy import DDPStrategy
from xinnovation.src.components.trainer.callbacks import Checkpoint
from xinnovation.src.components.trainer.loggers import TensorBoardLogger

def train(
    config: Union[str, Dict, Config],
    work_dir: str,
    resume_from: Optional[str] = None,
    gpu_ids: Optional[list] = None,
    seed: int = 42,
    validate: bool = True
):
    """训练流程
    
    Args:
        config (Union[str, Dict, Config]): 配置对象或配置文件路径
        work_dir (str): 工作目录
        resume_from (str, optional): 恢复训练的检查点路径
        gpu_ids (list, optional): GPU ID列表
        seed (int): 随机种子
        validate (bool): 是否进行验证
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
    
    # 构建数据集
    train_dataset = DATA.build(config['data']['train'])
    val_dataset = DATA.build(config['data']['val']) if validate else None
    
    # 构建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = None
    if validate and val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['test']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
    # 构建优化器
    optimizer = TRAINER.build(
        config['train']['optimizer'],
        params=model.parameters()
    )
    
    # 构建学习率调度器
    scheduler = TRAINER.build(
        config['train']['scheduler'],
        optimizer=optimizer
    )
    
    # 构建训练策略
    strategy = TRAINER.build(
        config['train'].get('strategy', dict(type='DDPStrategy')),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        work_dir=work_dir
    )
    
    # 构建回调
    callbacks = []
    for callback_cfg in config['train']['callbacks']:
        callbacks.append(TRAINER.build(callback_cfg))
        
    # 构建日志记录器
    loggers = []
    for logger_cfg in config['train']['loggers']:
        loggers.append(TRAINER.build(logger_cfg))
        
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
        
    # 设置训练参数
    strategy.setup(
        callbacks=callbacks,
        loggers=loggers,
        metrics=metrics,
        analyzers=analyzers,
        visualizers=visualizers,
        num_epochs=config['train']['epochs']
    )
    
    # 恢复训练
    if resume_from:
        strategy.load_checkpoint(resume_from)
        
    # 开始训练
    strategy.fit() 