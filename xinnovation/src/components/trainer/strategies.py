from xinnovation.src.core.registry import STRATEGIES
from lightning.pytorch.strategies import DDPStrategy


@STRATEGIES.register_module()
class DDPStrategy(DDPStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

