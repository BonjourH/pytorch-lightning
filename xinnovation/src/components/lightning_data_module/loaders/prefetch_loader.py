import torch
from torch.utils.data import DataLoader
from xinnovation.src.core.registry import LOADERS

@LOADERS.register_module()
class PrefetchLoader:
    """Prefetch data loader that loads next batch in background.
    
    Args:
        dataset: Dataset to load from
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        pin_memory (bool): Whether to pin memory
        prefetch_factor (int): Number of batches to prefetch
    """
    
    def __init__(
        self,
        dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        prefetch_factor=2
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        
        # Create data loader
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor
        )
        
        # Initialize iterator
        self.iter = None
        
    def __iter__(self):
        """Get iterator."""
        self.iter = iter(self.loader)
        return self
        
    def __next__(self):
        """Get next batch."""
        if self.iter is None:
            self.iter = iter(self.loader)
        return next(self.iter)
        
    def __len__(self):
        """Get number of batches."""
        return len(self.loader) 