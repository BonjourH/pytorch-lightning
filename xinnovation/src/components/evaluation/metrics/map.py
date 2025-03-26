import torch
import numpy as np
from xinnovation.src.core.registry import METRICS

@METRICS.register_module()
class MeanAP:
    """Mean Average Precision metric for object detection.
    
    Args:
        iou_threshold (float): IoU threshold for matching
        num_classes (int): Number of classes
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.5,
        num_classes: int = 80
    ):
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        
    def compute(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels):
        """Compute mean AP.
        
        Args:
            pred_boxes (torch.Tensor): Predicted boxes (N, 4)
            pred_scores (torch.Tensor): Predicted scores (N,)
            pred_labels (torch.Tensor): Predicted labels (N,)
            gt_boxes (torch.Tensor): Ground truth boxes (M, 4)
            gt_labels (torch.Tensor): Ground truth labels (M,)
            
        Returns:
            float: Mean AP across all classes
        """
        # Convert to numpy for easier computation
        pred_boxes = pred_boxes.cpu().numpy()
        pred_scores = pred_scores.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()
        gt_boxes = gt_boxes.cpu().numpy()
        gt_labels = gt_labels.cpu().numpy()
        
        # Compute AP for each class
        aps = []
        for c in range(self.num_classes):
            # Get predictions and ground truth for current class
            pred_mask = pred_labels == c
            gt_mask = gt_labels == c
            
            if not pred_mask.any() and not gt_mask.any():
                continue
                
            # Sort predictions by score
            scores = pred_scores[pred_mask]
            boxes = pred_boxes[pred_mask]
            sorted_idx = np.argsort(-scores)
            boxes = boxes[sorted_idx]
            
            # Compute IoU matrix
            gt_boxes_c = gt_boxes[gt_mask]
            if len(gt_boxes_c) == 0:
                continue
                
            iou_matrix = self._compute_iou_matrix(boxes, gt_boxes_c)
            
            # Compute precision and recall
            precision, recall = self._compute_pr(iou_matrix)
            
            # Compute AP
            ap = self._compute_ap(precision, recall)
            aps.append(ap)
            
        return np.mean(aps) if aps else 0.0
        
    def _compute_iou_matrix(self, boxes1, boxes2):
        """Compute IoU matrix between two sets of boxes."""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        iw = np.minimum(np.expand_dims(boxes1[:, 2], axis=1), boxes2[:, 2]) - \
             np.maximum(np.expand_dims(boxes1[:, 0], axis=1), boxes2[:, 0])
        ih = np.minimum(np.expand_dims(boxes1[:, 3], axis=1), boxes2[:, 3]) - \
             np.maximum(np.expand_dims(boxes1[:, 1], axis=1), boxes2[:, 1])
             
        iw = np.maximum(iw, 0)
        ih = np.maximum(ih, 0)
        
        intersection = iw * ih
        union = np.expand_dims(area1, axis=1) + area2 - intersection
        
        return intersection / np.maximum(union, 1e-6)
        
    def _compute_pr(self, iou_matrix):
        """Compute precision and recall."""
        num_pred = iou_matrix.shape[0]
        num_gt = iou_matrix.shape[1]
        
        # Match predictions to ground truth
        matched = np.zeros(num_gt, dtype=bool)
        tp = np.zeros(num_pred)
        
        for i in range(num_pred):
            ious = iou_matrix[i]
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]
            
            if max_iou >= self.iou_threshold and not matched[max_iou_idx]:
                tp[i] = 1
                matched[max_iou_idx] = True
                
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        precision = tp_cumsum / np.arange(1, num_pred + 1)
        recall = tp_cumsum / num_gt
        
        return precision, recall
        
    def _compute_ap(self, precision, recall):
        """Compute average precision."""
        # Add sentinel values
        precision = np.concatenate([precision, [0]])
        recall = np.concatenate([[0], recall])
        
        # Compute area under curve
        for i in range(precision.size - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])
            
        i = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])
        
        return ap 