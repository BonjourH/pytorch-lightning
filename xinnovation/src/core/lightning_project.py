from xinnovation.src.core.registry import LIGHTNING, LIGHTNING_MODULE, LIGHTNING_DATA_MODULE, TRAINER, DEPLOY
from xinnovation.src.components.deploy.deploy import Deployer
from typing import Dict
import lightning.pytorch as pl
from easydict import EasyDict as edict
import os
import torch

@LIGHTNING.register_module()
class LightningProject:
    def __init__(self, config: Dict):
        self.config = edict(config)
        
        self.model_module = LIGHTNING_MODULE.build(self.config.lightning_module)
        self.data_module = LIGHTNING_DATA_MODULE.build(self.config.lightning_data_module)
        self.trainer = TRAINER.build(self.config.lightning_trainer)
        self.deployer = DEPLOY.build(self.config.deploy)
        # self.evaluation_system = EvaluationCoordinator(config.evaluation)

        self.checkpoint_path = self._get_checkpoint_path()

    def _get_checkpoint_path(self):
        checkpoint_path = os.path.join(self.config.checkpoint_dir, 'last.ckpt')
        return checkpoint_path if os.path.exists(checkpoint_path) else None

    def train(self):
        if self.checkpoint_path is not None and self.config.resume:
            self.trainer.fit(self.model_module, self.data_module, ckpt_path=self.checkpoint_path)
        else:
            self.trainer.fit(self.model_module, self.data_module)

    def validate(self):
        if self.checkpoint_path is not None:
            self.trainer.validate(self.model_module, self.data_module, ckpt_path=self.checkpoint_path)
        else:
            self.trainer.validate(self.model_module, self.data_module)

    def predict(self):
        if self.checkpoint_path is None:
            raise ValueError("Checkpoint path is not set")
        self.trainer.predict(self.model_module, self.data_module, ckpt_path=self.checkpoint_path)

    def export(self, format: str='onnx', save_dir: str="/tmp"):
        model_name = (self.model_module.__class__.__name__).lower()
        
        # 确保检查点路径存在
        if self.checkpoint_path is None:
            raise ValueError("没有找到检查点文件，请先训练模型或指定检查点路径")
        
        # 将模型移到CPU上进行导出 - ONNX导出最好在CPU上进行
        model = self.model_module.__class__.load_from_checkpoint(self.checkpoint_path, map_location='cpu')
        model.eval()  # 确保模型处于评估模式
        model = model.cpu()  # 显式移动到CPU
                
        class ONNXWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, images):
                batch = {
                    "images": images  # 你 forward 接收的 key 是啥就写啥
                }
                return self.model(batch)
            
        wrapped_model = ONNXWrapper(model)
        
        # 加载数据
        self.data_module.setup("predict")
        predict_dataloader = self.data_module.predict_dataloader()
        sample_inputs = next(iter(predict_dataloader))
        
        # 确保输入数据也在CPU上
        if isinstance(sample_inputs, dict):
            sample_inputs_cpu = {}
            for k, v in sample_inputs.items():
                if isinstance(v, torch.Tensor):
                    sample_inputs_cpu[k] = v.cpu()
                else:
                    sample_inputs_cpu[k] = v
        elif isinstance(sample_inputs, torch.Tensor):
            sample_inputs_cpu = sample_inputs.cpu()
        else:
            sample_inputs_cpu = sample_inputs
        
        # 导出模型
        if format == 'onnx':
            self.deployer.convert_to_onnx(wrapped_model, sample_inputs_cpu, f"{save_dir}/{model_name}.onnx")
        elif format == 'torchscript':
            self.deployer.convert_to_torchscript(wrapped_model, sample_inputs_cpu, f"{save_dir}/{model_name}.pt")
        elif format == 'coreml':
            self.deployer.convert_to_coreml(wrapped_model, sample_inputs_cpu, f"{save_dir}/{model_name}.mlmodel")
        else:
            raise ValueError(f"Unsupported format: {format}")
