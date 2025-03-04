import torch
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np

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