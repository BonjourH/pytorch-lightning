from xinnovation.src.core.registry import LIGHTNING, LIGHTNING_MODULE, LIGHTNING_DATA_MODULE, TRAINER
from typing import Dict
import lightning.pytorch as pl
from easydict import EasyDict as edict
import os

@LIGHTNING.register_module()
class LightningProject:
    def __init__(self, config: Dict):
        self.config = edict(config)
        
        self.model_module = LIGHTNING_MODULE.build(self.config.lightning_module)
        self.data_module = LIGHTNING_DATA_MODULE.build(self.config.lightning_data_module)
        self.trainer = TRAINER.build(self.config.lightning_trainer)
        # self.deploy_engine = DeployController(config.deploy)
        # self.evaluation_system = EvaluationCoordinator(config.evaluation)

        self.checkpoint_path = self._get_checkpoint_path()

    def _get_checkpoint_path(self):
        checkpoint_path = os.path.join(self.config.checkpoint_dir, 'last.ckpt')
        if os.path.exists(checkpoint_path) and self.config.resume:
            return checkpoint_path
        else:
            return None

    def train(self):
        if self.checkpoint_path is not None:
            self.trainer.fit(self.model_module, self.data_module, ckpt_path=self.checkpoint_path)
        else:
            self.trainer.fit(self.model_module, self.data_module)

    def validate(self):
        if self.checkpoint_path is not None:
            self.trainer.validate(self.model_module, self.data_module, ckpt_path=self.checkpoint_path)
        else:
            self.trainer.validate(self.model_module, self.data_module)

    def predict(self, inputs):
        if self.checkpoint_path is not None:
            self.trainer.predict(self.model_module, self.data_module, inputs, ckpt_path=self.checkpoint_path)

    def export(self, format: str):
        pass

