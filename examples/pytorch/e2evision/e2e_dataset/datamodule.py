import os
import lightning as L
from typing import List, Optional
from torch.utils.data import DataLoader
from torchvision import transforms

from e2e_dataset.dataset import MultiFrameDataset, custom_collate_fn
from configs.config import DataConfig

class E2EPerceptionDataModule(L.LightningDataModule):
    """Lightning data module for end-to-end perception."""
    
    def __init__(self,
                 train_list: str,
                 val_list: str,
                 batch_size: int = 2,
                 num_workers: int = 4,
                 data_config: DataConfig = DataConfig()):
        super().__init__()
        self.save_hyperparameters(ignore=['camera_ids'])
        
        # Initialize datasets to None
        self.train_dataset: Optional[MultiFrameDataset] = None
        self.val_dataset: Optional[MultiFrameDataset] = None
    
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
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == "fit" or stage is None:
            # Read train and validation clip lists
            train_clips = self.read_clip_list(self.hparams.train_list)
            val_clips = self.read_clip_list(self.hparams.val_list)
            
            print(f"Found {len(train_clips)} training clips and {len(val_clips)} validation clips")
            
            # Create datasets
            self.train_dataset = MultiFrameDataset(
                clip_dirs=train_clips,
                config=self.hparams.data_config
            )
            
            self.val_dataset = MultiFrameDataset(
                clip_dirs=val_clips,
                config=self.hparams.data_config
            )
        
        elif stage == "validate":
            # Only setup validation dataset for validation stage
            if self.val_dataset is None:
                val_clips = self.read_clip_list(self.hparams.val_list)
                print(f"Found {len(val_clips)} validation clips")
                
                self.val_dataset = MultiFrameDataset(
                    clip_dirs=val_clips,
                    config=self.hparams.data_config
                )
        else:
            raise ValueError(f"Invalid stage: {stage}")
        print(f"Successfully setup datasets")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.data_config.shuffle,
            persistent_workers=self.hparams.data_config.persistent_workers,
            num_workers=self.hparams.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True
        ) 


if __name__ == "__main__":
    datamodule = E2EPerceptionDataModule(
        train_list="train_clips.txt",
        val_list="val_clips.txt",
        data_config=DataConfig()
    )
    datamodule.setup()