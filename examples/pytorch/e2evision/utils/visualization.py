import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
import cv2
from PIL import Image
import io
import wandb
from base import SourceCameraId, ObjectType

class Visualizer:
    """Visualization tools for model outputs and training progress."""
    
    @staticmethod
    def visualize_predictions(
        images: Dict[SourceCameraId, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
        max_images: int = 8
    ) -> Dict[str, wandb.Image]:
        """Visualize model predictions on images.
        
        Args:
            images: Dict[camera_id -> Tensor[B, T, C, H, W]]
            predictions: Dict containing model predictions
            targets: Optional ground truth
            max_images: Maximum number of images to visualize
            
        Returns:
            Dict of wandb.Image objects for logging
        """
        vis_dict = {}
        
        # Process each camera view
        for camera_id, image_tensor in images.items():
            B, T, C, H, W = image_tensor.shape
            num_vis = min(B, max_images)
            
            for b in range(num_vis):
                # Create figure with subplots for each timestep
                fig, axes = plt.subplots(1, T, figsize=(4*T, 4))
                if T == 1:
                    axes = [axes]
                
                for t, ax in enumerate(axes):
                    # Convert image to numpy
                    img = image_tensor[b, t].permute(1, 2, 0).cpu().numpy()
                    img = (img * 255).astype(np.uint8)
                    
                    # Draw image
                    ax.imshow(img)
                    
                    # Draw predictions
                    if 'traj_params' in predictions:
                        pred_trajs = predictions['traj_params'][b]
                        pred_types = predictions['type_logits'][b].argmax(dim=-1)
                        
                        for traj, obj_type in zip(pred_trajs, pred_types):
                            if traj[11] > 0.5:  # HAS_OBJECT flag
                                # Draw bounding box
                                bbox = Visualizer._get_2d_bbox(traj, camera_id)
                                if bbox is not None:
                                    rect = plt.Rectangle(
                                        (bbox[0], bbox[1]),
                                        bbox[2] - bbox[0],
                                        bbox[3] - bbox[1],
                                        fill=False,
                                        color='r'
                                    )
                                    ax.add_patch(rect)
                                    
                                    # Add label
                                    ax.text(
                                        bbox[0], bbox[1],
                                        f"{ObjectType(obj_type.item()).name}",
                                        color='r'
                                    )
                    
                    # Draw ground truth if available
                    if targets is not None and 'gt_trajectories' in targets:
                        gt_trajs = targets['gt_trajectories'][b]
                        gt_types = targets['gt_types'][b]
                        gt_masks = targets['gt_masks'][b]
                        
                        for traj, obj_type, mask in zip(gt_trajs, gt_types, gt_masks):
                            if mask > 0.5:
                                # Draw ground truth box
                                bbox = Visualizer._get_2d_bbox(traj, camera_id)
                                if bbox is not None:
                                    rect = plt.Rectangle(
                                        (bbox[0], bbox[1]),
                                        bbox[2] - bbox[0],
                                        bbox[3] - bbox[1],
                                        fill=False,
                                        color='g'
                                    )
                                    ax.add_patch(rect)
                    
                    ax.set_title(f"t={t}")
                    ax.axis('off')
                
                # Convert figure to wandb.Image
                vis_dict[f"{camera_id.name}_sample_{b}"] = wandb.Image(
                    Visualizer._fig2img(fig)
                )
                plt.close(fig)
        
        return vis_dict
    
    @staticmethod
    def visualize_attention(
        attention_weights: torch.Tensor,
        images: Dict[SourceCameraId, torch.Tensor],
        max_queries: int = 4
    ) -> Dict[str, wandb.Image]:
        """Visualize attention weights on images.
        
        Args:
            attention_weights: Tensor[B, num_queries, H*W]
            images: Dict[camera_id -> Tensor[B, T, C, H, W]]
            max_queries: Maximum number of queries to visualize
            
        Returns:
            Dict of wandb.Image objects for logging
        """
        vis_dict = {}
        B, Q, _ = attention_weights.shape
        num_queries = min(Q, max_queries)
        
        for camera_id, image_tensor in images.items():
            _, _, _, H, W = image_tensor.shape
            attention_map = attention_weights.view(B, Q, H, W)
            
            for b in range(B):
                fig, axes = plt.subplots(2, num_queries, figsize=(4*num_queries, 8))
                
                for q in range(num_queries):
                    # Original image
                    img = image_tensor[b, -1].permute(1, 2, 0).cpu().numpy()
                    axes[0, q].imshow(img)
                    axes[0, q].set_title(f"Query {q}")
                    axes[0, q].axis('off')
                    
                    # Attention map
                    att_map = attention_map[b, q].cpu().numpy()
                    axes[1, q].imshow(att_map, cmap='hot')
                    axes[1, q].set_title(f"Attention")
                    axes[1, q].axis('off')
                
                vis_dict[f"{camera_id.name}_attention_{b}"] = wandb.Image(
                    Visualizer._fig2img(fig)
                )
                plt.close(fig)
        
        return vis_dict
    
    @staticmethod
    def visualize_metrics(metrics: Dict[str, float]) -> Dict[str, Union[float, wandb.Image]]:
        """Visualize training metrics.
        
        Args:
            metrics: Dict of metric names and values
            
        Returns:
            Dict of visualizations for logging
        """
        vis_dict = {}
        
        # Add raw metrics
        vis_dict.update(metrics)
        
        # Create custom plots if needed
        # Example: Precision-Recall curve
        if 'precision' in metrics and 'recall' in metrics:
            # Create line plot using wandb
            data = [[metrics['recall'], metrics['precision']]]
            table = wandb.Table(data=data, columns=['recall', 'precision'])
            vis_dict['pr_curve'] = wandb.plot.scatter(
                table,
                'recall',
                'precision',
                title='Precision-Recall'
            )
        
        return vis_dict
    
    @staticmethod
    def _get_2d_bbox(traj: torch.Tensor, camera_id: SourceCameraId) -> Optional[np.ndarray]:
        """Project 3D trajectory to 2D bounding box in image space."""
        # Implementation depends on your projection method
        # Should return [x1, y1, x2, y2] or None if not visible
        return None
    
    @staticmethod
    def _fig2img(fig: plt.Figure) -> np.ndarray:
        """Convert matplotlib figure to numpy array."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        return np.array(img) 