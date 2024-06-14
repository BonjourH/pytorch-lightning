import os
import json
from PIL import Image
import pandas as pd
from dataclasses import dataclass, field, fields
from typing import List, Optional, Any, Union, Type, Dict, overload
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch, torchvision
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from enum import IntEnum
from tutorials.bev.base import *
import easydict

# -----------------------------
# 定义自定义数据集
# -----------------------------
class TrainingSample:
    # breif: a training sample contains k frames data
    def __init__(self, calibs: Dict[SourceCameraId, Dict]) -> None:
        self.image_paths: List[Dict[SourceCameraId, str]] = []
        self.calibration = calibs
        self.ego_states: List[Dict] = []
        self.obstacle_3D_states: List[List[Dict]] = []
    
    def add_frame_data(self, image_paths: Dict[SourceCameraId, str], ego_state: Dict, label: List[Dict]):
        self.image_paths.append(image_paths)
        self.ego_states.append(ego_state)
        self.obstacle_3D_states.append(label)
    
    
class MultiClipMultiFrameMultiCameraDataset(Dataset):
    def __init__(self, clip_dirs, camera_ids: List[SourceCameraId], transform=None, k=10):
        self.clip_dirs = clip_dirs
        self.camera_ids = camera_ids
        self.transform = transform
        self.k = k
        self.data = []

        for clip_dir in clip_dirs:
            # 1. load calibrations
            calibs = {}
            for camera in camera_ids:
                calib_file = os.path.join(clip_dir, f'calib/{camera.name.lower()}.json')
                with open(calib_file, 'r') as f:
                    calibs[camera] = json.load(f)
            # 2. load ego state && labels
            with open(os.path.join(clip_dir, 'ego_states.json'), 'r') as f:
                ego_states = json.load(f)
            with open(os.path.join(clip_dir, 'obstacle_labels.json'), 'r') as f:
                obstacle_labels = json.load(f)
            assert len(ego_states) == len(obstacle_labels), "ego_states and labels should have the same length but got {} and {}".format(len(ego_states), len(obstacle_labels))
            valid_sample_num = max(0, len(ego_states) - k + 1)
            # images
            for idx in range(valid_sample_num):
                ts = TrainingSample(calibs)
                # which contains k frames image paths, ego_states, labels and calibration
                for j in range(idx, idx+k):
                    timestamp = "{:.6f}".format(ego_states[j]['timestamp'])
                    img_paths = {}
                    for camera in camera_ids:
                        img_path = os.path.join(clip_dir, f'{camera.name.lower()}/{timestamp}.png')
                        assert os.path.exists(img_path), f"image path {img_path} not exists"
                        img_paths[camera] = img_path
                    ts.add_frame_data(img_paths, ego_states[j], obstacle_labels[j])
                self.data.append(ts)
            print(f"clip {clip_dir} has {valid_sample_num} valid samples")
        print(f"total {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame = self.data[idx]
        ret = {}
        # 1. load images
        imgs: Dict[SourceCameraId, torch.Tensor] = {} # torch.Tensor: [k, C, H, W]
        for camera in self.camera_ids:
            img_paths = [frame.image_paths[i][camera] for i in range(self.k)]
            imgs[camera] = torch.stack([self.transform(Image.open(img_path)) for img_path in img_paths])
        ret['images'] = imgs 
        # imgs[1].shape, torch.Size([10, 3, 540, 960])
        # 2. load calibration
        calibs: Dict[SourceCameraId, torch.Tensor] = {} # torch.Tensor: [k, 20]
        for camera in self.camera_ids:
            calibs[camera] = torch.tensor(convert_calib_dict_to_vec(frame.calibration[camera]), dtype=torch.float32)
        ret['calibration'] = calibs
        # 3. load ego state
        ego_states: List[List[float]] = [convert_ego_state_dict_to_vec(state) for state in frame.ego_states]
        ret['ego_states'] = torch.tensor(ego_states, dtype=torch.float32)
        # 4. load labels
        max_obstacle_num = max([len(frame.obstacle_3D_states[i]) for i in range(self.k)])
        obstacle_states: List[List[List[float]]] = []
        for i in range(self.k):
            labels = []
            for j in range(len(frame.obstacle_3D_states[i])):
                labels.append(convert_obstable_3D_state_to_vec(frame.obstacle_3D_states[i][j]))
            for j in range(len(frame.obstacle_3D_states[i]), max_obstacle_num):
                labels.append([-1 for _ in range(ObstacleStateIndex.END_INDEX)])
            obstacle_states.append(labels)
        ret['labels'] = torch.tensor(obstacle_states, dtype=torch.float32)
        return ret

# 定义数据预处理
data_transform = transforms.Compose([
    transforms.Resize((416, 800)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 参数
k = 10  # 使用的帧数
source_camera_ids = [SourceCameraId.FRONT_LEFT_CAMERA, SourceCameraId.FRONT_RIGHT_CAMERA, 
                     SourceCameraId.FRONT_CENTER_CAMERA, SourceCameraId.SIDE_LEFT_CAMERA, 
                     SourceCameraId.SIDE_RIGHT_CAMERA, SourceCameraId.REAR_LEFT_CAMERA, 
                     SourceCameraId.REAR_RIGHT_CAMERA]  # 使用的相机

ego_state_dim = 3  # 假设自车状态是一个3维向量

# 假设多个片段目录存储在以下文件夹中
clip_dirs = ['/home/xinjian/Code/auto_labeling_pipeline/labeling_info/2_20231219T122348_pdb-l4e-c0011_0_0to8_0.716']
            #  '/home/xinjian/Code/auto_labeling_pipeline/labeling_info/5_20240223T161731_pdb-l4e-b0001_4_159to169_0.171',
            #  '/home/xinjian/Code/auto_labeling_pipeline/labeling_info/8_20240117T084829_pdb-l4e-b0005_11_855to885_0.563']

# 实例化数据集
dataset = MultiClipMultiFrameMultiCameraDataset(clip_dirs=clip_dirs, camera_ids=source_camera_ids, transform=data_transform, k=k)

# debug
# res = dataset.__getitem__(0)
# res['images'][1].shape torch.Size([10, 3, 540, 960])
# res['calibration'][1].shape torch.Size([20])
# res['ego_states'].shape torch.Size([10, 12])
# res['labels'].shape torch.Size([10, k, 16])
# import pdb; pdb.set_trace()

# 实例化数据加载器
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# debug
# Create an iterator from the DataLoader
# data_iter = iter(data_loader)

# Get the next batch of data
# batch_data = next(data_iter)
# batch_data['images'][1].shape, torch.Size([4, 10, 3, 416, 800])
# import pdb; pdb.set_trace()
# exit()

# -----------------------------
# 定义模型
# -----------------------------

class ImageFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=128):
        super(ImageFeatureExtractor, self).__init__()
        # Load the pretrained ResNet18 model
        resnet18 = torchvision.models.resnet18(pretrained=True)
        # Remove the fully connected layer, Global Average Pooling, and the last Max Pooling layer
        self.img_feature_extractor = nn.Sequential(
            *list(resnet18.children())[:-2],
            nn.Conv2d(512, feature_dim, kernel_size=1), # Optional: Add a Conv layer to reduce dimension if required
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Forward pass through the feature extractor
        x = self.img_feature_extractor(x)
        return x
    
    
class NaiveBEVModel(L.LightningModule):
    def __init__(self, camera_ids: List[SourceCameraId], k: int, img_feature_dim: int=128, bev_feature_dim: int=128, output_dim: int=16):
        super().__init__()
        self.camera_ids = camera_ids
        self.front_encoder = ImageFeatureExtractor(img_feature_dim)
        self.side_encoder = ImageFeatureExtractor(img_feature_dim)
        self.rear_encoder = ImageFeatureExtractor(img_feature_dim)
    
    def extact_img_features(self, imgs, encoder):
        # imgs: torch.Tensor [batch_size, k, C, H, W]
        # encoder: ImageFeatureExtractor
        batch_size, k, C, H, W = imgs.shape
        imgs = imgs.view(-1, C, H, W)
        features = encoder(imgs)
        return features.view(batch_size, k, -1)
        
    
    def forward(self, imgs, calibs, ego_states):
        # 1. extract 2D images features
        img_features = {}
        for cam_id, img in imgs.items():
            if is_front_camera(cam_id):
                img_features[cam_id] = self.extact_img_features(img, self.front_encoder)
            elif is_side_camera(cam_id):
                img_features[cam_id] = self.extact_img_features(img, self.side_encoder)
            elif is_rear_camera(cam_id):
                img_features[cam_id] = self.extact_img_features(img, self.rear_encoder)
        # 2. project 2D features to BEV
        # write the code here
        
        # 3. concatenate all features
        # 4. predict object position and speed
        return output

    def training_step(self, batch, batch_idx):
        # 1. extract 2D images features
        output = self.forward(batch['images'], batch['calibration'], batch['ego_states'])
        
        
    def validation_step(self, batch, batch_idx):
        frames, ego_state, target = batch
        output = self.forward(frames, ego_state)
        loss = F.mse_loss(output, target)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# 实例化模型
model = NaiveBEVModel()

# -----------------------------
# 训练模型
# -----------------------------
# 初始化 WandbLogger
logger = WandbLogger(project="multi_frame_multi_camera_object_detection", name="object_position_speed")

# 初始化 ModelCheckpoint 回调
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/',
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min'
)

# 实例化 Trainer
trainer = L.Trainer(logger=logger, callbacks=[checkpoint_callback], max_epochs=5)

# 训练模型
trainer.fit(model, data_loader)
