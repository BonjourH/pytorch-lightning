import torch
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import TQDMProgressBar, RichProgressBar
from tqdm import tqdm


class FilteredProgressBar(TQDMProgressBar):
    """自定义进度条，只显示特定的指标"""
    
    def __init__(self, refresh_rate: int = 1, process_position: int = 0, metrics_to_display=None):
        super().__init__(refresh_rate, process_position)
        # 定义要显示的指标列表
        self.metrics_to_display = metrics_to_display or [
            'train/loss_step', 
            'train/loss_epoch',
            'train/layer_6_loss_exist_epoch',
            'val/loss',
            'epoch',
            'step'
        ]
    
    def init_train_tqdm(self):
        return tqdm(
            desc="Training",
            total=self.total_train_batches,
            leave=True,       # 保留所有进度条
            dynamic_ncols=True,
            unit="batch",
            disable=self.is_disabled,
            position=0,       # 固定位置
            postfix=self.metrics_to_display  # 显示指标
        )
    
    def get_metrics(self, trainer, pl_module):
        # 获取所有指标
        items = super().get_metrics(trainer, pl_module)
        # 过滤指标，只保留我们想要显示的
        filtered_items = {}
        
        basic_metrics = ['epoch', 'step']
        
        # 首先添加基本指标
        for basic_metric in basic_metrics:
            if basic_metric in items:
                filtered_items[basic_metric] = items[basic_metric]
        
        # 然后添加配置中指定的指标
        for k, v in items.items():
            if k in self.metrics_to_display:
                if k in basic_metrics:
                    continue
                filtered_items[k] = v
        
        return filtered_items


class E2EPerceptionWandbLogger(WandbLogger):
    def __init__(self, *args, keys_to_log=None, use_optional_metrics=False, **kwargs):
        super().__init__(*args, **kwargs)
        # 定义要记录的指标列表
        self.keys_to_log = keys_to_log or [
            'train/loss_epoch', 
            'train/layer_6_loss_exist_epoch'
        ]
        self.use_optional_metrics = use_optional_metrics
    
    def log_metrics(self, metrics, step=None):
        # 只记录特定的键
        if self.use_optional_metrics:
            metrics_to_log = {key: metrics[key] for key in self.keys_to_log if key in metrics}
        else:
            metrics_to_log = metrics
        super().log_metrics(metrics_to_log, step)


def compute_trajectory_metrics(pred_trajs: torch.Tensor,
                            gt_trajs: torch.Tensor,
                            gt_masks: torch.Tensor,
                            distance_threshold: float = 2.0) -> Dict[str, float]:
    """Compute trajectory prediction metrics.
    
    Args:
        pred_trajs: Tensor[B, N, TrajParamIndex.END_OF_INDEX] - Predicted trajectories
        gt_trajs: Tensor[B, N, TrajParamIndex.END_OF_INDEX] - Ground truth trajectories
        gt_masks: Tensor[B, N] - Ground truth masks (1 for valid objects)
        distance_threshold: Distance threshold for success
        
    Returns:
        Dict containing metrics:
            - ade: Average displacement error
            - fde: Final displacement error
            - success_rate: Percentage of successful predictions
    """
    # Compute displacement error
    error = torch.norm(pred_trajs[..., :2] - gt_trajs[..., :2], dim=-1)  # [B, N]
    
    # Mask out invalid objects
    error = error * gt_masks
    
    # Compute metrics
    ade = error.sum() / (gt_masks.sum() + 1e-6)
    fde = error[..., -1].sum() / (gt_masks.sum() + 1e-6)
    
    # Compute success rate
    success = (error <= distance_threshold).float() * gt_masks
    success_rate = success.sum() / (gt_masks.sum() + 1e-6)
    
    return {
        'ade': ade.item(),
        'fde': fde.item(),
        'success_rate': success_rate.item()
    }

def compute_classification_metrics(pred_logits: torch.Tensor,
                                gt_types: torch.Tensor,
                                gt_masks: torch.Tensor) -> Dict[str, float]:
    """Compute classification metrics.
    
    Args:
        pred_logits: Tensor[B, N, num_classes] - Predicted class logits
        gt_types: Tensor[B, N] - Ground truth types
        gt_masks: Tensor[B, N] - Ground truth masks (1 for valid objects)
        
    Returns:
        Dict containing metrics:
            - accuracy: Classification accuracy
            - precision: Precision score
            - recall: Recall score
            - f1: F1 score
    """
    # Get predictions
    pred_types = torch.argmax(pred_logits, dim=-1)  # [B, N]
    
    # Compute metrics only on valid objects
    correct = (pred_types == gt_types).float() * gt_masks
    accuracy = correct.sum() / (gt_masks.sum() + 1e-6)
    
    # Compute per-class metrics
    num_classes = pred_logits.shape[-1]
    precision = []
    recall = []
    
    for c in range(num_classes):
        pred_c = (pred_types == c).float() * gt_masks
        gt_c = (gt_types == c).float() * gt_masks
        
        tp = (pred_c * gt_c).sum()
        fp = (pred_c * (1 - gt_c)).sum()
        fn = ((1 - pred_c) * gt_c).sum()
        
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        
        precision.append(prec)
        recall.append(rec)
    
    precision = torch.stack(precision).mean()
    recall = torch.stack(recall).mean()
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }

def evaluate_predictions(outputs: Dict,
                       targets: Dict) -> Dict[str, float]:
    """Evaluate model predictions.
    
    Args:
        outputs: Dict containing model outputs
        targets: Dict containing ground truth
        
    Returns:
        Dict containing all metrics
    """
    # Get final predictions
    pred_trajs = outputs[-1]['traj_params']
    pred_logits = outputs[-1]['type_logits']
    
    # Get ground truth
    gt_trajs = targets['gt_trajectories']
    gt_types = targets['gt_types']
    gt_masks = targets['gt_masks']
    
    # Compute metrics
    traj_metrics = compute_trajectory_metrics(pred_trajs, gt_trajs, gt_masks)
    cls_metrics = compute_classification_metrics(pred_logits, gt_types, gt_masks)
    
    # Combine metrics
    metrics = {}
    metrics.update(traj_metrics)
    metrics.update(cls_metrics)
    
    return metrics 