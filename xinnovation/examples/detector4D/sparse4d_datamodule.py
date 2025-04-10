import os
import lightning as L
from typing import List, Optional, Dict
from torch.utils.data import DataLoader
from torchvision import transforms

from xinnovation.examples.detector4D.sparse4d_dataset import Sparse4DMultiFrameDataset, mutli_frame_collate_fn, CameraGroupConfig
from xinnovation.src.core import DATASETS, LIGHTNING_DATA_MODULE


@LIGHTNING_DATA_MODULE.register_module()
class Sparse4DDataModule(L.LightningDataModule):
    """Lightning data module for end-to-end perception."""
    
    def __init__(self,
                 train_list: str,
                 val_list: str,
                 predict_list: str,
                 test_list: str,
                 batch_size: int,
                 num_workers: int,
                 sequence_length: int,
                 shuffle: bool,
                 persistent_workers: bool,
                 pin_memory: bool,
                 camera_groups: List[CameraGroupConfig],
                 xrel_range: List[float],
                 yrel_range: List[float],
                 sliding_window_size: int = 20,
                 sliding_window_stride: int = 2):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize datasets to None
        self.train_dataset: Optional[Sparse4DMultiFrameDataset] = None
        self.val_dataset: Optional[Sparse4DMultiFrameDataset] = None
        self.predict_dataset: Optional[Sparse4DMultiFrameDataset] = None
        self.test_dataset: Optional[Sparse4DMultiFrameDataset] = None

    @classmethod
    def read_clip_list(cls, list_file: str) -> List[str]:
        """Read clip paths from a txt file."""
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"Clip list file not found: {list_file}")
            
        with open(list_file, 'r') as f:
            clips = [line.strip() for line in f.readlines() if line.strip()]
            
        # Verify all paths exist
        for clip_path in clips:
            if not os.path.exists(clip_path):
                raise FileNotFoundError(f"Clip directory not found: {clip_path}")
                
        return clips
    
    def _get_sparse4d_dataset(self, clip_dirs: List[str]):
        return Sparse4DMultiFrameDataset(
            clip_dirs=clip_dirs,
            sequence_length=self.hparams.sequence_length,
            camera_groups=self.hparams.camera_groups,
            xrel_range=self.hparams.xrel_range,
            yrel_range=self.hparams.yrel_range,
            sliding_window_size=self.hparams.sliding_window_size,
            sliding_window_stride=self.hparams.sliding_window_stride
        )
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == "fit" or stage is None:
            # Read train and validation clip lists
            train_clips = self.read_clip_list(self.hparams.train_list)
            val_clips = self.read_clip_list(self.hparams.val_list)
            
            print(f"Found {len(train_clips)} training clips and {len(val_clips)} validation clips")
            
            # Create datasets
            self.train_dataset = self._get_sparse4d_dataset(train_clips)
            self.val_dataset = self._get_sparse4d_dataset(val_clips)
        
        elif stage == "validate":
            # Only setup validation dataset for validation stage
            if self.val_dataset is None:
                val_clips = self.read_clip_list(self.hparams.val_list)
                print(f"Found {len(val_clips)} validation clips")
                
                self.val_dataset = self._get_sparse4d_dataset(val_clips)
        elif stage == "predict":
            # Only setup predict dataset for predict stage
            if self.predict_dataset is None:
                predict_clips = self.read_clip_list(self.hparams.predict_list)
                print(f"Found {len(predict_clips)} predict clips")
                
                self.predict_dataset = self._get_sparse4d_dataset(predict_clips)
        elif stage == "test":
            # Only setup test dataset for test stage
            if self.test_dataset is None:
                test_clips = self.read_clip_list(self.hparams.test_list)
                print(f"Found {len(test_clips)} test clips")
                
                self.test_dataset = self._get_sparse4d_dataset(test_clips)
        else:
            raise ValueError(f"Invalid stage: {stage}")
        print(f"Successfully setup datasets")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle,
            persistent_workers=self.hparams.persistent_workers,
            num_workers=self.hparams.num_workers,
            collate_fn=mutli_frame_collate_fn,
            pin_memory=self.hparams.pin_memory
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=mutli_frame_collate_fn,
            pin_memory=self.hparams.pin_memory
        ) 
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=mutli_frame_collate_fn,
            pin_memory=self.hparams.pin_memory
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=mutli_frame_collate_fn,
            pin_memory=self.hparams.pin_memory
        )


if __name__ == "__main__":
    cur_dir = os.getcwd()
    train_list = os.path.join(cur_dir, "train_clips.txt")
    val_list = os.path.join(cur_dir, "val_clips.txt")
    predict_list = os.path.join(cur_dir, "predict_clips.txt")
    datamodule = Sparse4DDataModule(
        train_list=train_list,
        val_list=val_list,
        predict_list=predict_list,
        sequence_length=10,
        camera_groups=[CameraGroupConfig.front_stereo_camera_group(), CameraGroupConfig.short_focal_length_camera_group(), CameraGroupConfig.rear_camera_group()],
        batch_size=1,
        num_workers=1,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True
    )
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    for batch in train_dataloader:
        print(batch)
        break
