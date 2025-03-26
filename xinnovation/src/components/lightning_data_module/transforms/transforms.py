import torch
import torchvision.transforms as T
from PIL import Image
from xinnovation.src.core.registry import TRANSFORMS

@TRANSFORMS.register_module()
class Resize:
    """Resize image and boxes.
    
    Args:
        size (tuple): Target size (height, width)
    """
    
    def __init__(self, size):
        self.size = size
        
    def __call__(self, data):
        img = data['img']
        h, w = self.size
        
        # Resize image
        img = img.resize((w, h), Image.BILINEAR)
        
        # Resize boxes if present
        if 'gt_bboxes' in data:
            boxes = data['gt_bboxes']
            scale_x = w / img.width
            scale_y = h / img.height
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            data['gt_bboxes'] = boxes
            
        data['img'] = img
        return data

@TRANSFORMS.register_module()
class RandomFlip:
    """Randomly flip image and boxes.
    
    Args:
        prob (float): Probability of flipping
        direction (str): Direction to flip ('horizontal' or 'vertical')
    """
    
    def __init__(self, prob=0.5, direction='horizontal'):
        self.prob = prob
        self.direction = direction
        
    def __call__(self, data):
        if torch.rand(1) < self.prob:
            img = data['img']
            w, h = img.size
            
            # Flip image
            if self.direction == 'horizontal':
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                
            # Flip boxes if present
            if 'gt_bboxes' in data:
                boxes = data['gt_bboxes']
                if self.direction == 'horizontal':
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                else:
                    boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                data['gt_bboxes'] = boxes
                
            data['img'] = img
        return data

@TRANSFORMS.register_module()
class Normalize:
    """Normalize image with mean and std.
    
    Args:
        mean (list): Mean values for each channel
        std (list): Std values for each channel
    """
    
    def __init__(self, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
        self.mean = mean
        self.std = std
        
    def __call__(self, data):
        img = data['img']
        img = T.ToTensor()(img)
        img = T.Normalize(mean=self.mean, std=self.std)(img)
        data['img'] = img
        return data

@TRANSFORMS.register_module()
class ColorJitter:
    """Randomly adjust image color.
    
    Args:
        brightness (float): Brightness adjustment range
        contrast (float): Contrast adjustment range
        saturation (float): Saturation adjustment range
        hue (float): Hue adjustment range
    """
    
    def __init__(
        self,
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1
    ):
        self.transform = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        
    def __call__(self, data):
        data['img'] = self.transform(data['img'])
        return data 