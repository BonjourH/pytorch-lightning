from xinnovation.src.core.registry import CALLBACKS
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar, Callback
from tqdm import tqdm
from typing import List

__all__ = ["CheckpointCallback", "EarlyStoppingCallback", "LearningRateMonitorCallback", "FilteredProgressBarCallback"]

@CALLBACKS.register_module()
class CheckpointCallback(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@CALLBACKS.register_module()
class EarlyStoppingCallback(EarlyStopping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@CALLBACKS.register_module()
class LearningRateMonitorCallback(LearningRateMonitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@CALLBACKS.register_module()
class FilteredProgressBarCallback(TQDMProgressBar):
    def __init__(self, metrics_to_display: List[str], refresh_rate: int = 1, process_position: int = 0):
        super().__init__(refresh_rate, process_position)
        # 定义要显示的指标列表
        self.metrics_to_display = metrics_to_display
    
    def init_train_tqdm(self):
        return tqdm(
            desc="Training",
            total=self.total_train_batches,
            leave=True,       # 保留所有进度条
            dynamic_ncols=True,
            unit="batch",
            disable=self.is_disabled,
            position=0,       # 固定位置
            postfix=self.metrics_to_display  # 显示指标
        )
    
    def get_metrics(self, trainer, pl_module):
        # 获取所有指标
        items = super().get_metrics(trainer, pl_module)
        # 过滤指标，只保留我们想要显示的
        filtered_items = {}
        
        basic_metrics = ['epoch', 'step']
        
        # 首先添加基本指标
        for basic_metric in basic_metrics:
            if basic_metric in items:
                filtered_items[basic_metric] = items[basic_metric]
        
        # 然后添加配置中指定的指标
        for k, v in items.items():
            if k in self.metrics_to_display:
                if k in basic_metrics:
                    continue
                filtered_items[k] = v
        
        return filtered_items
    
if __name__ == "__main__":
    cfg = dict(type="FilteredProgressBarCallback", metrics_to_display=["loss", "acc"])
    callback = CALLBACKS.build(cfg)
    print(callback)