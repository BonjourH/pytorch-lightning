import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from base import ObstacleTrajectory, ObjectType, tensor_to_trajectory

class TrajectoryEvaluator:
    """Evaluation metrics for trajectory prediction."""
    
    @staticmethod
    def compute_metrics(
        predictions: List[ObstacleTrajectory],
        targets: List[ObstacleTrajectory],
        max_distance: float = 50.0,
        iou_threshold: float = 0.5
    ) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            predictions: List of predicted trajectories
            targets: List of ground truth trajectories
            max_distance: Maximum distance for considering matches
            iou_threshold: IoU threshold for considering matches
            
        Returns:
            Dict of metrics
        """
        metrics = {}
        
        # Match predictions to ground truth
        matches = TrajectoryEvaluator._match_trajectories(
            predictions, targets, max_distance, iou_threshold
        )
        
        # Compute detection metrics
        tp = len(matches)
        fp = len(predictions) - tp
        fn = len(targets) - tp
        
        metrics['precision'] = tp / (tp + fp) if tp + fp > 0 else 0.0
        metrics['recall'] = tp / (tp + fn) if tp + fn > 0 else 0.0
        metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (
            metrics['precision'] + metrics['recall']
        ) if metrics['precision'] + metrics['recall'] > 0 else 0.0
        
        # Compute trajectory metrics
        if matches:
            position_errors = []
            velocity_errors = []
            yaw_errors = []
            type_accuracy = 0
            
            for pred_idx, gt_idx in matches:
                pred = predictions[pred_idx]
                gt = targets[gt_idx]
                
                # Position error
                position_error = np.linalg.norm(pred.position - gt.position)
                position_errors.append(position_error)
                
                # Velocity error
                velocity_error = np.linalg.norm(pred.velocity - gt.velocity)
                velocity_errors.append(velocity_error)
                
                # Yaw error (considering circular nature)
                yaw_error = abs(pred.yaw - gt.yaw)
                yaw_error = min(yaw_error, 2 * np.pi - yaw_error)
                yaw_errors.append(yaw_error)
                
                # Type accuracy
                if pred.object_type == gt.object_type:
                    type_accuracy += 1
            
            metrics['mean_position_error'] = np.mean(position_errors)
            metrics['mean_velocity_error'] = np.mean(velocity_errors)
            metrics['mean_yaw_error'] = np.mean(yaw_errors)
            metrics['type_accuracy'] = type_accuracy / len(matches)
        
        return metrics
    
    @staticmethod
    def _match_trajectories(
        predictions: List[ObstacleTrajectory],
        targets: List[ObstacleTrajectory],
        max_distance: float,
        iou_threshold: float
    ) -> List[Tuple[int, int]]:
        """Match predicted trajectories to ground truth using Hungarian algorithm."""
        from scipy.optimize import linear_sum_assignment
        
        if not predictions or not targets:
            return []
        
        # Compute cost matrix
        cost_matrix = np.zeros((len(predictions), len(targets)))
        
        for i, pred in enumerate(predictions):
            for j, gt in enumerate(targets):
                # Distance cost
                distance = np.linalg.norm(pred.position - gt.position)
                if distance > max_distance:
                    cost_matrix[i, j] = float('inf')
                    continue
                
                # IoU cost
                iou = TrajectoryEvaluator._compute_box_iou(pred, gt)
                if iou < iou_threshold:
                    cost_matrix[i, j] = float('inf')
                    continue
                
                cost_matrix[i, j] = distance * (1 - iou)
        
        # Find optimal matching
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        
        # Filter invalid matches
        matches = []
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            if cost_matrix[pred_idx, gt_idx] != float('inf'):
                matches.append((pred_idx, gt_idx))
        
        return matches
    
    @staticmethod
    def _compute_box_iou(traj1: ObstacleTrajectory, traj2: ObstacleTrajectory) -> float:
        """Compute IoU between two 3D bounding boxes."""
        # Get box corners
        corners1 = traj1.corners()
        corners2 = traj2.corners()
        
        # Project to BEV
        corners1_bev = corners1[:, :2]  # Use only x,y coordinates
        corners2_bev = corners2[:, :2]
        
        # Compute intersection area
        intersection = TrajectoryEvaluator._polygon_intersection_area(
            corners1_bev, corners2_bev
        )
        
        # Compute union area
        area1 = traj1.length * traj1.width
        area2 = traj2.length * traj2.width
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def _polygon_intersection_area(poly1: np.ndarray, poly2: np.ndarray) -> float:
        """Compute intersection area of two convex polygons."""
        from shapely.geometry import Polygon
        
        p1 = Polygon(poly1)
        p2 = Polygon(poly2)
        
        if p1.intersects(p2):
            return p1.intersection(p2).area
        return 0.0

class TemporalEvaluator:
    """Evaluation metrics for temporal consistency."""
    
    @staticmethod
    def compute_metrics(
        trajectory_sequences: List[List[ObstacleTrajectory]],
        dt: float = 0.1
    ) -> Dict[str, float]:
        """Compute temporal consistency metrics.
        
        Args:
            trajectory_sequences: List of trajectory lists for each timestep
            dt: Time step between frames
            
        Returns:
            Dict of metrics
        """
        metrics = {}
        
        # Track ID consistency
        id_switches = 0
        track_fragments = 0
        
        # Motion smoothness
        position_smoothness = []
        velocity_smoothness = []
        yaw_smoothness = []
        
        for t in range(1, len(trajectory_sequences)):
            prev_trajs = trajectory_sequences[t-1]
            curr_trajs = trajectory_sequences[t]
            
            # Match trajectories between frames
            matches = TrajectoryEvaluator._match_trajectories(
                curr_trajs, prev_trajs, max_distance=5.0, iou_threshold=0.5
            )
            
            # Count ID switches and fragments
            matched_prev = set(gt_idx for _, gt_idx in matches)
            matched_curr = set(pred_idx for pred_idx, _ in matches)
            
            id_switches += len(matches)  # Different IDs for same object
            track_fragments += (len(prev_trajs) - len(matched_prev))  # Lost tracks
            
            # Compute motion smoothness
            for pred_idx, prev_idx in matches:
                curr_traj = curr_trajs[pred_idx]
                prev_traj = prev_trajs[prev_idx]
                
                # Position smoothness (acceleration)
                pos_diff = (curr_traj.position - prev_traj.position) / dt
                pos_smoothness = np.linalg.norm(pos_diff - prev_traj.velocity)
                position_smoothness.append(pos_smoothness)
                
                # Velocity smoothness (jerk)
                vel_diff = (curr_traj.velocity - prev_traj.velocity) / dt
                vel_smoothness = np.linalg.norm(vel_diff - prev_traj.acceleration)
                velocity_smoothness.append(vel_smoothness)
                
                # Yaw smoothness
                yaw_diff = abs(curr_traj.yaw - prev_traj.yaw)
                yaw_diff = min(yaw_diff, 2 * np.pi - yaw_diff)
                yaw_smoothness.append(yaw_diff / dt)
        
        # Compute final metrics
        metrics['id_switches'] = id_switches
        metrics['track_fragments'] = track_fragments
        metrics['mean_position_smoothness'] = np.mean(position_smoothness) if position_smoothness else 0.0
        metrics['mean_velocity_smoothness'] = np.mean(velocity_smoothness) if velocity_smoothness else 0.0
        metrics['mean_yaw_smoothness'] = np.mean(yaw_smoothness) if yaw_smoothness else 0.0
        
        return metrics 