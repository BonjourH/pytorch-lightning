from xinnovation.src.core.registry import STRATEGIES
from lightning.pytorch.strategies import DDPStrategy


@STRATEGIES.register_module()
class LightningDDPStrategy(DDPStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

