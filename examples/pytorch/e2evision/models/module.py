import lightning as L
import torch
import torch.nn as nn
import cv2
import numpy as np
import os, sys, json
from typing import Dict, List, Optional, Union, Tuple
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from base import SourceCameraId, TrajParamIndex
from models.network import E2EPerceptionNet
from models.loss import TrajectoryLoss
from configs.config import Config
from e2e_dataset.dataset import TrainingSample
from utils.pose_transform import project_points_to_image
from utils.math_utils import generate_unit_cube_points, generate_bbox_corners_points, sample_bbox_edge_points

class E2EPerceptionModule(L.LightningModule):
    """Lightning module for end-to-end perception."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])
        self.config = config
        # Create network
        self.net = E2EPerceptionNet(config.model, config.data, config.training.use_checkpoint)
        
        # Create loss function
        self.criterion = TrajectoryLoss(config.loss)
        
        # Initialize validation metrics
        self.val_step_outputs = []
        
        # Initialize predict metrics
        self.predict_step_outputs = []
        
        # for visualization debug
        if self.config.logging.visualize_intermediate_results:
            self.register_buffer('bbox_edge_points', sample_bbox_edge_points(1000))
        
    def forward(self, batch: Dict) -> List[Dict]:
        return self.net(batch)
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Create optimizer
        optimizer = Adam(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Create scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.training.learning_rate,
            epochs=self.config.training.max_epochs,
            steps_per_epoch=self.trainer.estimated_stepping_batches // self.config.training.max_epochs,
            pct_start=self.config.training.pct_start,
            div_factor=self.config.training.div_factor,
            final_div_factor=self.config.training.final_div_factor,
            three_phase=True
        )
        
        onc_cycle_config = {
            "scheduler": scheduler,
            "interval": "step",  # OneCycleLR is updated every step
            "frequency": 1
        }
        
        # scheduler = CosineAnnealingLR(
        #     optimizer,
        #     T_max=self.config.training.max_epochs,
        #     eta_min=1e-6
        # )
        
        # cosine annealing_config = {
        #     "optimizer": optimizer,
        #     #monitor="val/loss",
        #     "interval": "epoch",
        #     "frequency": 1
        # }
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": onc_cycle_config
        }
    
    def _render_trajs_on_imgs(self, 
                      trajs: torch.Tensor, 
                      camera_ids: List[SourceCameraId],
                      calibrations: torch.Tensor,
                      ego_states: torch.Tensor, 
                      imgs_dict: Dict[SourceCameraId, torch.Tensor], 
                      color: torch.Tensor=torch.tensor([255.0, 0.0, 0.0])) -> Tuple[Dict[SourceCameraId, torch.Tensor], Dict[SourceCameraId, np.ndarray]]:
        """
        Render trajectories on images.
        Args:
            trajs: [B, N, TrajParamIndex.END_OF_INDEX]
            camera_ids: List[SourceCameraId]
            calibrations: torch.Tensor[B, Ncams, CameraParamIndex.END_OF_INDEX]
            ego_states: torch.Tensor[B, EgoStateParamIndex.END_OF_INDEX]
            imgs_dict: Dict[SourceCameraId, torch.Tensor[B, T, 3, H, W]]
        Returns:
            imgs_dict: Dict[SourceCameraId, torch.Tensor[B, T, 3, H, W]]
            concat_imgs: Dict[SourceCameraId, np.ndarray]
        """
        
        pixels, _ = project_points_to_image(trajs, calibrations, ego_states, self.bbox_edge_points)
        B, C, N, T, P, _ = pixels.shape # [B, C, N, T, P, 2]
        
        # disable pixels of false positive trajectories
        traj_fp_mask = trajs[..., TrajParamIndex.HAS_OBJECT] < 0.5 # [B, N]
        traj_fp_mask = traj_fp_mask.unsqueeze(1).unsqueeze(3).unsqueeze(4).expand(-1, C, -1, T, P)
        pixels[traj_fp_mask, :] = -1

        concat_imgs: Dict[SourceCameraId, np.ndarray] = {}
        
        # plot pixels on images
        color = color.to(self.device)
        for cam_idx, camera_id in enumerate(camera_ids):
            img_sequence = imgs_dict[camera_id] # [B, T, H, W, 3]
            H, W = img_sequence.shape[2:4]
            
            tmp_pixel = pixels[:, cam_idx, ...] # [B, N, T, P, 2]
            tmp_pixel = tmp_pixel.permute(0, 2, 1, 3, 4) # [B, T, N, P, 2]
            tmp_pixel = tmp_pixel.reshape(B, T, N * P, 2) # [B, T, N * P, 2]
            tmp_pixel[:, :, :, 0] *= W
            tmp_pixel[:, :, :, 1] *= H
            tmp_invalid_mask = torch.logical_or(
                torch.logical_or(tmp_pixel[:, :, :, 0] < 0, tmp_pixel[:, :, :, 1] < 0),
                torch.logical_or(tmp_pixel[:, :, :, 0] > W - 1, tmp_pixel[:, :, :, 1] > H - 1)
            )
            
            tmp_pixel[tmp_invalid_mask, :] = -1
            
            h_coords = tmp_pixel[:, :, :, 1].long()
            w_coords = tmp_pixel[:, :, :, 0].long()
            mask = torch.ones_like(img_sequence).to(self.device) # [B, T, H, W, 3]
            
            # 利用PyTorch的高级索引功能，通过向量化操作快速修改像素
            batch_indices = torch.arange(B, device=self.device)[:, None, None].expand(-1, T, N*P)
            time_indices = torch.arange(T, device=self.device)[None, :, None].expand(B, -1, N*P)
            r = (self.config.logging.point_radius + 1) // 2
            r = max(r, 1)
            for i in range(-r, r):
                h_indices = h_coords + i
                h_indices = torch.clamp(h_indices, 0, H - 1)
                for j in range(-r, r):
                    w_indices = w_coords + j
                    w_indices = torch.clamp(w_indices, 0, W - 1)
                    mask[batch_indices, time_indices, h_indices, w_indices, :] = 0
            img_sequence = img_sequence * mask + color * (1 - mask)
            imgs_dict[camera_id] = img_sequence
            
            # save concat imgs
            cimg = img_sequence.permute(0, 2, 1, 3, 4) # [B, H, T, W, 3]
            cimg = cimg.reshape(B * H, T * W, 3) # [B * H, T * W, 3]
            cimg = cimg.cpu().numpy() # [B * H, T * W, 3]
            cimg = cimg.astype(np.uint8)
            concat_imgs[camera_id] = cimg
            
            # NOTE: img_sequence[batch_indices, time_indices, tmp_pixel[:, :, :, 1], tmp_pixel[:, :, :, 0], :] = color
            # will report error: img_sequence[batch_indices, time_indices, tmp_pixel[:, :, :, 1], tmp_pixel[:, :, :, 0], :] = color
            # RuntimeError: linearIndex.numel()*sliceSize*nElemBefore == expandedValue.numel() INTERNAL ASSERT FAILED at 
            # "/pytorch/aten/src/ATen/native/cuda/Indexing.cu":548, please report a bug to PyTorch. number of flattened indices 
            # did not match number of elements in the value tensor: 15360 vs 9

            debug = False
            if debug:
                cv2.imwrite('img2.png', cimg)
                cv2.imshow('img', cimg)
                cv2.waitKey(0)
                exit(0)
    
        return imgs_dict, concat_imgs
    
        
    def visualize_gt_trajs(self, batch: Dict):
        """Visualize ground truth trajectories."""
        # 1. read images
        imgs_dict = TrainingSample.read_seqeuntial_images_to_tensor(batch['image_paths'], self.device)
        # imgs_dict: Dict[SourceCameraId, torch.Tensor[B, T, C, H, W]]
        
        # 2. render gt trajs
        imgs_dict, concat_imgs = self._render_trajs_on_imgs(batch['trajs'], 
                                                    batch['camera_ids'],
                                                    batch['calibrations'],
                                                    batch['ego_states'], 
                                                    imgs_dict,
                                                    color=torch.tensor([0.0, 255.0, 0.0]))
       
        # 3. save images
        save_dir = os.path.join(self.config.logging.visualize_intermediate_results_dir, 'gt')
        os.makedirs(save_dir, exist_ok=True)
        for camera_id in imgs_dict.keys():
            cv2.imwrite(os.path.join(save_dir, f'gt_{camera_id}.png'), concat_imgs[camera_id])
        
    def visualize_pred_trajs(self, outputs: List[Dict]):
        """Visualize predicted trajectories."""
        pass
    
    def on_train_start(self):
        """On train start."""
        print("On train start")
        # save config file
        config_path = os.path.join(self.config.logging.log_dir, "train_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Training step."""
        # # Forward pass
        outputs, _ = self(batch)
        
        # # Compute loss
        loss_dict = self.criterion(outputs, batch['trajs'])
        
        # # Log losses
        for name, value in loss_dict.items():
            self.log(f"train/{name}", value, on_step=True, on_epoch=True, prog_bar=True)
            
        if self.config.logging.visualize_intermediate_results:
            print(f'Begin to visualize gt and pred trajs for training step {batch_idx}')
            # 1. read images
            imgs_dict = TrainingSample.read_seqeuntial_images_to_tensor(batch['image_paths'], self.device)
            # 2. render gt trajs
            imgs_dict, concat_imgs = self._render_trajs_on_imgs(batch['trajs'], 
                                                    batch['camera_ids'],
                                                    batch['calibrations'],
                                                    batch['ego_states'], 
                                                    imgs_dict,
                                                    color=torch.tensor([0.0, 255.0, 0.0]))
            # 3. render pred trajs
            # imgs_dict, concat_imgs = self._render_trajs_on_imgs(outputs[-1]['trajs'], 
            #                                         batch['camera_ids'],
            #                                         batch['calibrations'],
            #                                         batch['ego_states'], 
            #                                         imgs_dict,
            #                                         color=torch.tensor([0.0, 0.0, 255.0]))
            
            # 4. save images
            save_dir = os.path.join(self.config.logging.visualize_intermediate_results_dir)
            os.makedirs(save_dir, exist_ok=True)
            for camera_id in concat_imgs.keys():
                img_name = f'{camera_id.name}_{batch_idx}.png'
                cv2.imwrite(os.path.join(save_dir, img_name), concat_imgs[camera_id])

        return loss_dict
    
    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Validation step."""
        # Forward pass
        outputs, _ = self(batch)
        
        # Compute loss
        loss_dict = self.criterion(outputs, batch['trajs'])
        
        # Store outputs for epoch end
        self.val_step_outputs.append({
            'loss_dict': loss_dict,
            'outputs': outputs[-1],  # Only keep final predictions
            'targets': batch
        })
        
        # Log losses
        for name, value in loss_dict.items():
            self.log(f"val/{name}", value, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss_dict
    
    def on_validation_epoch_end(self):
        """Compute validation metrics at epoch end."""
        if not self.val_step_outputs:
            print("No validation outputs to process")
            return
        
        print(f"Processing {len(self.val_step_outputs)} validation outputs")
        
        # Aggregate predictions and targets
        all_preds = []
        all_targets = []
        
        for i, output in enumerate(self.val_step_outputs):
            print(f"Processing validation output {i+1}/{len(self.val_step_outputs)}")
            
            # 这是一个张量，不是字典
            pred_trajs = output['outputs']
            gt_trajs = output['targets']['trajs']
            
            print(f"  Predictions shape: {pred_trajs.shape}")
            print(f"  Targets shape: {gt_trajs.shape}")
            
            # Filter valid predictions and targets
            valid_mask_preds = torch.sigmoid(pred_trajs[..., TrajParamIndex.HAS_OBJECT]) > 0.5  # HAS_OBJECT flag
            valid_mask_targets = gt_trajs[..., TrajParamIndex.HAS_OBJECT] > 0.5
            
            print(f"  Valid predictions: {valid_mask_preds.sum().item()}")
            print(f"  Valid targets: {valid_mask_targets.sum().item()}")
            
            valid_preds = pred_trajs[valid_mask_preds]
            valid_targets = gt_trajs[valid_mask_targets]
            
            all_preds.extend(valid_preds)
            all_targets.extend(valid_targets)
        
        print(f"Total valid predictions: {len(all_preds)}")
        print(f"Total valid targets: {len(all_targets)}")
        
        # Compute metrics
        metrics = self._compute_metrics(all_preds, all_targets)
        
        # Log metrics
        for name, value in metrics.items():
            self.log(f"val/{name}", value, on_epoch=True)
        
        # Clear outputs
        self.val_step_outputs.clear()
    
    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Test step."""
        return self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch: Dict, batch_idx: int, dataloader_idx: int = 0) -> Dict:
        """Prediction step."""
        outputs, _ = self(batch)
        self.predict_step_outputs.append(outputs)
        return outputs
    
    def on_predict_epoch_end(self):
        """On predict epoch end."""
        print("On predict epoch end")
        
    def on_predict_start(self):
        """On predict start."""
        print("On predict start")
        # save config file
        os.makedirs(self.config.predict.output_dir, exist_ok=True)
        config_path = os.path.join(self.config.predict.output_dir, "predict_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
    def on_predict_end(self):
        """On predict end."""
        print("On predict end")
        
    def _compute_metrics(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]) -> Dict:
        """Compute evaluation metrics."""
        metrics = {}
        
        # TODO: Implement detailed metrics computation
        # - Position error
        # - Velocity error
        # - Classification accuracy
        # - Detection metrics (precision, recall, F1)
        
        return metrics 