import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Any, List, Tuple

import lightning.pytorch as L
from xinnovation.src.core import LIGHTNING_MODULE, SourceCameraId, TrajParamIndex
from xinnovation.src.components.lightning_module import LightningDetector
from xinnovation.src.utils.math_utils import sample_bbox_edge_points
from xinnovation.src.utils.pose_transform import project_points_to_image
from .sparse4d_detector import Sparse4DDetector
from .sparse4d_loss import Sparse4DLossWithDAC
from .sparse4d_dataset import TrainingSample, CAMERA_ID_LIST
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob


__all__ = ["Sparse4DModule"]

@LIGHTNING_MODULE.register_module()
class Sparse4DModule(LightningDetector):
    
    def __init__(self, detector: Dict, loss: Dict, optimizer: Dict, scheduler: Dict, **kwargs):
        super().__init__(detector=detector, loss=loss, optimizer=optimizer, scheduler=scheduler, **kwargs)

        assert "debug_config" in kwargs, "debug_config is required"
        self.debug_config = kwargs["debug_config"]

        # Initialize validation metrics
        self.val_step_outputs = []
        
        # Initialize predict metrics
        self.predict_step_outputs = []
        
        # for visualization
        if self.debug_config.visualize_intermediate_results:
            self.register_buffer("bbox_edge_points", sample_bbox_edge_points(50))
        
    def forward(self, batch) -> List[Dict]:
        return self.detector(batch)
    
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
            calibrations: torch.Tensor[B, C, CameraParamIndex.END_OF_INDEX]
            ego_states: torch.Tensor[B, T, EgoStateParamIndex.END_OF_INDEX]
            imgs_dict: Dict[SourceCameraId, torch.Tensor[B, T, 3, H, W]]
        Returns:
            imgs_dict: Dict[SourceCameraId, torch.Tensor[B, T, 3, H, W]]
            concat_imgs: Dict[SourceCameraId, np.ndarray]
        """
        B, T, _ = ego_states.shape
        
        pixels = project_points_to_image(trajs, calibrations, ego_states, self.bbox_edge_points) # pixels: torch.Tensor[B*T, N, C, P, 2]
        _, N, C, P, _ = pixels.shape
        pixels = pixels.view(B, T, N, C, P, 2)
        
        # disable pixels of false positive trajectories
        traj_fp_mask = trajs[..., TrajParamIndex.HAS_OBJECT].sigmoid() < self.debug_config.pred_traj_threshold # [B, N]
        traj_fp_mask = traj_fp_mask.unsqueeze(1).unsqueeze(3).unsqueeze(4).expand(-1, T, -1, C, P)
        pixels[traj_fp_mask, :] = -1

        concat_imgs: Dict[SourceCameraId, np.ndarray] = {}
        
        # plot pixels on images
        color = color.to(self.device)
        for cam_idx, camera_id in enumerate(camera_ids):
            img_sequence = imgs_dict[camera_id] # [B, T, H, W, 3]
            H, W = img_sequence.shape[2:4]

            tmp_pixel = pixels[:, :, :, cam_idx, ...] # [B, T, N, P, 2]
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
            r = (self.debug_config.point_radius + 1) // 2
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
    
    def on_train_batch_start(self, batch, batch_idx):
        """On train batch start."""
        pass

    def on_train_epoch_start(self):
        """On train epoch start."""
        self.criterion.update_epoch(self.current_epoch)
            
    def on_train_start(self):
        """On train start."""
        print("On train start")
        # 确保重置匹配历史记录
        self.criterion.reset_matching_history()

        # create log dir
        os.makedirs(self.debug_config.log_dir, exist_ok=True)

        # save config file
        config_path = os.path.join(self.debug_config.log_dir, "train_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.hparams, f, indent=2)

    def _save_matching_history(self):
        """Save the matching history."""
        print("Visualizing matching history...")
        # 获取匹配历史记录
        matching_history = self.criterion.matching_history
        
        if not matching_history:
            print("No matching history to visualize.")
            return
        
        # 创建保存可视化结果的目录
        vis_dir = os.path.join(self.debug_config.log_dir, "matching_visualization")
        os.makedirs(vis_dir, exist_ok=True)
        
        # matching_history的结构是 Dict[int, List[Tuple[int, int]]]
        # 键是layer_idx，值是(pred_idx, gt_idx)匹配对列表
        # 注意：pred_idx和gt_idx可能是numpy数组而不是简单的整数
        
        # 首先保存原始匹配历史数据为CSV和JSON格式
        for layer_idx, matches in matching_history.items():
            # 保存为CSV
            csv_path = os.path.join(vis_dir, f'layer_{layer_idx}_matches.csv')
            with open(csv_path, 'w') as f:
                f.write("step,pred_idx,gt_idx\n")
                for step, match_pair in enumerate(matches):
                    pred_idx, gt_idx = match_pair
                    f.write(f"{step},{pred_idx},{gt_idx}\n")
            
            # 保存为JSON
            json_path = os.path.join(vis_dir, f'layer_{layer_idx}_matches.json')
            with open(json_path, 'w') as f:
                serializable_matches = []
                for match_pair in matches:
                    pred_idx, gt_idx = match_pair
                    # 处理numpy数组 - 转换为列表
                    if hasattr(pred_idx, 'tolist'):
                        pred_idx = pred_idx.tolist()
                    if hasattr(gt_idx, 'tolist'):
                        gt_idx = gt_idx.tolist()
                    serializable_matches.append([pred_idx, gt_idx])
                json.dump(serializable_matches, f, indent=2)
        
        # 获取所有层索引
        layer_indices = sorted(matching_history.keys())
        print(f"Found data for {len(layer_indices)} decoder layers")
        
        # 对每个层创建折线图可视化
        for layer_idx in layer_indices:
            # 获取该层的所有匹配记录
            matches = matching_history[layer_idx]
            print(f"Layer {layer_idx}: Found {len(matches)} match records")
            
            # 创建折线图
            plt.figure(figsize=(14, 8))
            
            # 收集每个GT索引的匹配信息
            gt_to_query_over_time = {}
            
            for step, (pred_idx, gt_idx) in enumerate(matches):
                gt_idx = gt_idx.tolist()
                pred_idx = pred_idx.tolist()
                for gt_idx_i, pred_idx_i in zip(gt_idx, pred_idx):
                    # 将匹配添加到对应GT的历史记录中
                    if gt_idx_i not in gt_to_query_over_time:
                        gt_to_query_over_time[gt_idx_i] = []
                    
                    gt_to_query_over_time[gt_idx_i].append(pred_idx_i)
            
            # 使用不同颜色绘制每个GT索引的匹配轨迹
            colors = plt.cm.rainbow(np.linspace(0, 1, len(gt_to_query_over_time)))
            color_idx = 0
            
            for gt_idx, query_over_time in gt_to_query_over_time.items():
                steps = range(len(query_over_time))
                
                plt.plot(steps, query_over_time, 'o-', linewidth=1.5, 
                        label=f'GT {gt_idx}', 
                        color=colors[color_idx % len(colors)],
                        markersize=4)
                color_idx += 1
            
            plt.grid(True)
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('Matched Query Index', fontsize=12)
            plt.title(f'Query Assignment Changes Over Training (Layer {layer_idx})', 
                      fontsize=14, fontweight='bold')
            
            # 保存图像
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'layer_{layer_idx}_matching_lines.png'), dpi=300)
            plt.close()
        
        print(f"Matching visualization completed. Results saved to {vis_dir}")

    def _generate_pred_trajs_video(self, mode: str="train"):
        """Generate the video of the predicted trajectories."""
        if not self.debug_config.visualize_intermediate_results:
            return
        print("Generating the video of the predicted trajectories...")
        trajs_dir = os.path.join(self.debug_config.visualize_intermediate_results_dir, mode)
        if not os.path.exists(trajs_dir):
            print(f"No predicted trajectories to visualize for {mode} mode")
            return
        
        img_files = glob.glob(os.path.join(trajs_dir, "*.png"))
        epoch_interval = self.debug_config.render_trajs_interval if mode == "train" else 1

        for camera_id in CAMERA_ID_LIST:
            camera_name = camera_id.name
            camera_img_files = [file for file in img_files if camera_name in file]
            camera_img_files.sort()
            
            # Save frames to a temporary directory
            temp_dir = os.path.join(trajs_dir, f"temp_{camera_name}")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Copy and rename frames sequentially for ffmpeg
            for i, img_file in enumerate(camera_img_files):
                img = cv2.imread(img_file)
                if img is not None:
                    frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                    # plot "Epoch xxx" on the top middle of the image
                    epoch = i * epoch_interval
                    cv2.putText(img, f"Epoch {epoch}", (img.shape[1] // 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imwrite(frame_path, img)
            
            # Use ffmpeg to create the video
            video_path = os.path.join(self.debug_config.visualize_intermediate_results_dir, f"{mode}_{camera_name}.mp4")
            ffmpeg_cmd = f"ffmpeg -y -framerate 10 -i {temp_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {video_path}"
            os.system(ffmpeg_cmd)
            
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir)

    def on_train_end(self):
        """On train end, visualize the matching history."""
        self._save_matching_history()
        self._generate_pred_trajs_video(mode="train")
        self._generate_pred_trajs_video(mode="val")
        
    def on_validation_end(self):
        """On validation end, visualize the matching history."""
        pass
        
    def visualize_init_trajs(self, batch: Dict):
        """Visualize initial trajectories."""
        # 1. read images
        imgs_dict = TrainingSample.read_seqeuntial_images_to_tensor(batch['image_paths'], self.device)
        # 2. render init trajs
        init_trajs = self.detector.get_init_trajs(batch['ego_states'].shape[0])
        imgs_dict, concat_imgs = self._render_trajs_on_imgs(init_trajs, 
                                                batch['camera_ids'],
                                                batch['calibrations'],
                                                batch['ego_states'], 
                                                imgs_dict,
                                                color=torch.tensor(self.debug_config.init_color))
        
        # 3. save images
        save_dir = os.path.join(self.debug_config.visualize_intermediate_results_dir, "init_trajs")
        os.makedirs(save_dir, exist_ok=True)
        for camera_id in concat_imgs.keys():
            img_name = f'Init_trajs_{camera_id.name}.png'
            cv2.imwrite(os.path.join(save_dir, img_name), concat_imgs[camera_id])

    def save_intermediate_results(self, batch: Dict, batch_idx: int, trajs: torch.Tensor, mode: str="train"):
        """
        Save intermediate results for visualization, only save the first batch for training and validation.
        Args:
            batch: Dict
            batch_idx: int
            trajs: torch.Tensor, take the trajs of the last decoder layer
        """
        if not (self.debug_config.visualize_intermediate_results and batch_idx == 0):
            return
        if mode == "train" and self.current_epoch % self.debug_config.render_trajs_interval != 0:
            return
        # 1. read images
        imgs_dict = TrainingSample.read_seqeuntial_images_to_tensor(batch['image_paths'], self.device)
        # 2. render gt trajs
        if self.debug_config.render_gt_trajs:
            imgs_dict, concat_imgs = self._render_trajs_on_imgs(batch['trajs'], 
                                                    batch['camera_ids'],
                                                    batch['calibrations'],
                                                    batch['ego_states'], 
                                                    imgs_dict,
                                                    color=torch.tensor(self.debug_config.gt_color))
        # 3. visualize pred trajs
        if self.debug_config.render_pred_trajs:
            imgs_dict, concat_imgs = self._render_trajs_on_imgs(trajs, 
                                                    batch['camera_ids'],
                                                    batch['calibrations'],
                                                    batch['ego_states'], 
                                                    imgs_dict,
                                                    color=torch.tensor(self.debug_config.pred_color))
        
        # 4. save images
        save_dir = os.path.join(self.debug_config.visualize_intermediate_results_dir, mode)
        os.makedirs(save_dir, exist_ok=True)
        epoch_str = f"{self.current_epoch:05d}"
        for camera_id in concat_imgs.keys():
            img_name = f'epoch_{epoch_str}_{camera_id.name}.png'
            cv2.imwrite(os.path.join(save_dir, img_name), concat_imgs[camera_id])
        
    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Training step."""
        # Forward pass
        outputs, c_outputs, quality = self(batch)
        
        self.save_intermediate_results(batch, batch_idx, outputs[-1], mode="train")
        # Compute loss
        loss_dict = self.criterion(batch['trajs'], outputs, c_outputs)
        
        # # Log losses
        for name, value in loss_dict.items():
            self.log(f"train/{name}", value, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch['trajs'].shape[0])
            
        return loss_dict
    
    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Validation step."""
        # Forward pass
        outputs, c_outputs, quality = self(batch)

        self.save_intermediate_results(batch, batch_idx, outputs[-1], mode="val")
        
        # Compute loss
        loss_dict = self.criterion(batch['trajs'], outputs, c_outputs)
        
        # Store outputs for epoch end
        self.val_step_outputs.append({
            'loss_dict': loss_dict,
            'outputs': outputs[-1],  # Only keep final predictions
            'targets': batch
        })
        
        # Log losses
        for name, value in loss_dict.items():
            self.log(f"val/{name}", value, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch['trajs'].shape[0])
        
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
            valid_mask_preds = torch.sigmoid(pred_trajs[..., TrajParamIndex.HAS_OBJECT]) > self.debug_config.pred_traj_threshold  # HAS_OBJECT flag
            valid_mask_targets = gt_trajs[..., TrajParamIndex.HAS_OBJECT] > self.debug_config.pred_traj_threshold
            
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
        outputs, _, _ = self(batch)
        self.predict_step_outputs.append(outputs)
        return outputs
    
    def on_predict_epoch_end(self):
        """On predict epoch end."""
        print("On predict epoch end")
        
    def on_predict_start(self):
        """On predict start."""
        print("On predict start")
        # save config file
        os.makedirs(self.debug_config.log_dir, exist_ok=True)
        os.makedirs(self.debug_config.predict_dir, exist_ok=True)
        config_path = os.path.join(self.debug_config.predict_dir, "predict_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.hparams, f, indent=2)
        
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
        