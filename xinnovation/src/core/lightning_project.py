from xinnovation.src.core.registry import LIGHTNING, LIGHTNING_MODULE, LIGHTNING_DATA_MODULE, TRAINER
from typing import Dict
import lightning.pytorch as pl
from easydict import EasyDict as edict

@LIGHTNING.register_module()
class LightningProject:
    def __init__(self, config: Dict):
        self.config = edict(config)
        
        self.model_module = LIGHTNING_MODULE.build(self.config.lightning_module)
        self.data_module = LIGHTNING_DATA_MODULE.build(self.config.lightning_data_module)
        self.trainer = TRAINER.build(self.config.lightning_trainer)
        # self.deploy_engine = DeployController(config.deploy)
        # self.evaluation_system = EvaluationCoordinator(config.evaluation)

    def train(self):
        pass

    def predict(self, inputs):
        pass

    def export(self, format: str):
        pass