import os
from torch.utils.tensorboard import SummaryWriter
from xinnovation.src.core.registry import LOGGERS

@LOGGERS.register_module()
class TensorBoardLogger:
    """TensorBoard logger for training metrics.
    
    Args:
        log_dir (str): Directory to save logs
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
    def log_metrics(self, metrics: dict, step: int):
        """Log metrics to TensorBoard.
        
        Args:
            metrics (dict): Dictionary of metric names and values
            step (int): Current step/epoch
        """
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
            
    def log_model_graph(self, model, input_shape=(1, 3, 224, 224)):
        """Log model graph to TensorBoard.
        
        Args:
            model: PyTorch model
            input_shape (tuple): Shape of input tensor
        """
        from torch.utils.tensorboard import SummaryWriter
        dummy_input = torch.randn(input_shape)
        self.writer.add_graph(model, dummy_input)
        self.writer.close()
        
    def log_images(self, images: dict, step: int):
        """Log images to TensorBoard.
        
        Args:
            images (dict): Dictionary of image names and tensors
            step (int): Current step/epoch
        """
        for name, image in images.items():
            self.writer.add_image(name, image, step)
            
    def close(self):
        """Close the writer."""
        self.writer.close() 