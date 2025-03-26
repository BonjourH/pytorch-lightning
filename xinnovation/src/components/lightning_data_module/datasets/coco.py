import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from xinnovation.src.core.registry import DATASETS

@DATASETS.register_module()
class COCODataset(Dataset):
    """COCO dataset for object detection.
    
    Args:
        ann_file (str): Path to annotation file
        img_prefix (str): Path to image directory
        transforms (list): List of transforms to apply
        filter_empty_gt (bool): Whether to filter empty ground truth
        test_mode (bool): Whether in test mode
    """
    
    def __init__(
        self,
        ann_file: str,
        img_prefix: str,
        transforms: list = None,
        filter_empty_gt: bool = True,
        test_mode: bool = False
    ):
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.transforms = transforms
        self.filter_empty_gt = filter_empty_gt
        self.test_mode = test_mode
        
        # Load annotations
        self.coco = self._load_annotations()
        self.ids = list(self.coco.imgs.keys())
        
        # Filter empty annotations
        if filter_empty_gt:
            self.ids = self._filter_empty_gt()
            
    def _load_annotations(self):
        """Load COCO annotations."""
        with open(self.ann_file, 'r') as f:
            return json.load(f)
            
    def _filter_empty_gt(self):
        """Filter images with empty ground truth."""
        ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                ids.append(img_id)
        return ids
        
    def __len__(self):
        """Get dataset length."""
        return len(self.ids)
        
    def __getitem__(self, idx):
        """Get data item.
        
        Args:
            idx (int): Index of data item
            
        Returns:
            dict: Data item
        """
        img_id = self.ids[idx]
        
        # Load image
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.img_prefix, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        ann_info = self.coco.loadAnns(ann_ids)
        
        # Convert annotations to format
        gt_bboxes = []
        gt_labels = []
        for ann in ann_info:
            x1, y1, w, h = ann['bbox']
            gt_bboxes.append([x1, y1, x1 + w, y1 + h])
            gt_labels.append(ann['category_id'])
            
        data = {
            'img': img,
            'gt_bboxes': torch.tensor(gt_bboxes, dtype=torch.float32),
            'gt_labels': torch.tensor(gt_labels, dtype=torch.long),
            'img_id': img_id
        }
        
        # Apply transforms
        if self.transforms is not None:
            for t in self.transforms:
                data = t(data)
                
        return data 