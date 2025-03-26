from xinnovation.src.core.registry import LIGHTNING, LIGHTNING_MODULE, DATA, TRAINER
from typing import Dict
import lightning.pytorch as pl
from easydict import EasyDict as edict

@LIGHTNING.register_module()
class LightningProject:
    def __init__(self, config: Dict):
        self.config = edict(config)
        
        self.model_module = LIGHTNING_MODULE.build(self.config.lightning_module)
        # self.data_module = DATA.build(self.config.data)
        # self.trainer = TRAINER.build(self.config.trainer)
        # self.deploy_engine = DeployController(config.deploy)
        # self.evaluation_system = EvaluationCoordinator(config.evaluation)

    def train(self):
        pass

    def predict(self, inputs):
        pass

    def export(self, format: str):
        pass