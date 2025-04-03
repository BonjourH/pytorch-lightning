from xinnovation.src.core.registry import LOGGERS
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger, CSVLogger
from typing import List

__all__ = ["LightningWandbLogger", "LightningTensorBoardLogger", "LightningCSVLogger"]


@LOGGERS.register_module()
class LightningWandbLogger(WandbLogger):
    def __init__(self, project: str, name: str, keys_to_log: List[str] = [], use_optional_metrics: bool = False, *args, **kwargs):
        super().__init__(project=project, name=name, *args, **kwargs)
        self.keys_to_log = keys_to_log
        self.use_optional_metrics = use_optional_metrics and len(keys_to_log) > 0
        
    def log_metrics(self, metrics: dict, step: int):
        if self.use_optional_metrics:
            metrics = {k: v for k, v in metrics.items() if k in self.keys_to_log}
        super().log_metrics(metrics, step)
        
    # def log_model_graph(self, model, input_shape=(1, 3, 224, 224)):
    #     """Log model graph to WandB.
        
    #     Args:
    #         model: PyTorch model
    #         input_shape (tuple): Shape of input tensor
    #     """
    #     super().log_model_graph(model, input_shape)
        
    # def log_images(self, images: dict, step: int):
    #     """Log images to WandB.
        
    #     Args:
    #         images (dict): Dictionary of image names and tensors
    #         step (int): Current step/epoch
    #     """
    #     super().log_images(images, step)
        
    # def close(self):
    #     super().close() 


@LOGGERS.register_module()
class LightningTensorBoardLogger(TensorBoardLogger):
    def __init__(self, save_dir: str, name: str, *args, **kwargs):
        super().__init__(save_dir=save_dir, name=name, *args, **kwargs)


@LOGGERS.register_module()
class LightningCSVLogger(CSVLogger):
    def __init__(self, save_dir: str, name: str, *args, **kwargs):
        super().__init__(save_dir=save_dir, name=name, *args, **kwargs)


if __name__ == "__main__":
    cfg = dict(type="LightningWandbLogger", project="test", name="test")
    wandb_logger = LOGGERS.build(cfg)
    cfg = dict(type="LightningTensorBoardLogger", save_dir="test", name="test")
    tb_logger = LOGGERS.build(cfg)
    cfg = dict(type="LightningCSVLogger", save_dir="test", name="test")
    csv_logger = LOGGERS.build(cfg)