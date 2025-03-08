import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import torch
import lightning as L
from torch.utils.data import DataLoader

# 自定义模块
from base import tensor_to_trajectory, TrajParamIndex
from e2e_dataset.dataset import MultiFrameDataset, custom_collate_fn
from e2e_dataset.datamodule import E2EPerceptionDataModule
from configs.config import get_config, Config
from models.module import E2EPerceptionModule

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Lightning-based Perception Inference')
    parser.add_argument('--config_file', type=str, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--test_list', type=str, help='Path to txt file containing test clip paths')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Inference batch size')
    
    parser.add_argument('--config-override', nargs='+', action='append', 
                        help='Override config values. Format: section.key=value')
    return parser.parse_args()

class PredictionWriter(L.Callback):
    """自定义回调用于保存预测结果"""
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.predictions = []
        
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """收集每个batch的预测结果"""
        # 转换张量为轨迹对象
        # outputs: List[Tensor[batch_size, query_num, TrajParamIndex.END_OF_INDEX]]
        
        valid_trajs = []
        last_layer_outputs = outputs[-1]
        for i in range(last_layer_outputs.shape[0]):
            debug = True
            if debug:
                batch_traj_vecs = last_layer_outputs[i]
                sorted_indices = torch.argsort(batch_traj_vecs[:, TrajParamIndex.HAS_OBJECT], descending=True)
                batch_traj_vecs = batch_traj_vecs[sorted_indices]
                print(f"\n query idx {sorted_indices[0]} has largest confidence of having object: {batch_traj_vecs[0][TrajParamIndex.HAS_OBJECT]:.2f},"
                      f"\n position: {batch_traj_vecs[0][TrajParamIndex.X]:.2f}, {batch_traj_vecs[0][TrajParamIndex.Y]:.2f}, {batch_traj_vecs[0][TrajParamIndex.Z]:.2f}"
                      f"\n dimension: {batch_traj_vecs[0][TrajParamIndex.LENGTH]:.2f}, {batch_traj_vecs[0][TrajParamIndex.WIDTH]:.2f}, {batch_traj_vecs[0][TrajParamIndex.HEIGHT]:.2f}"
                      f"\n velocity: {batch_traj_vecs[0][TrajParamIndex.VX]:.2f}, {batch_traj_vecs[0][TrajParamIndex.VY]:.2f},"
                      f"\n acceleration: {batch_traj_vecs[0][TrajParamIndex.AX]:.3f}, {batch_traj_vecs[0][TrajParamIndex.AY]:.3f}")
                
            for j in range(last_layer_outputs.shape[1]):
                traj_vec = last_layer_outputs[i][j]
                if traj_vec[TrajParamIndex.HAS_OBJECT] > 0.5:
                    valid_trajs.append(tensor_to_trajectory(traj_vec))
        
        print(f"This batch has {len(valid_trajs)} valid predictions")
        
        self.predictions.append(valid_trajs)
    
    def on_predict_end(self, trainer, pl_module):
        """保存所有预测结果"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "predictions.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.predictions, f, indent=2)
            
        print(f"\n✅ Predictions saved to {output_path}")

def main():
    args = parse_args()
    config = get_config(args)
    
    assert args.checkpoint, f'{args.checkpoint} is not found'
    
    model = E2EPerceptionModule.load_from_checkpoint(args.checkpoint, config=config)
    model.eval()
    
    # 准备数据集
    test_list = E2EPerceptionDataModule.read_clip_list(args.test_list)
    dataset = MultiFrameDataset(clip_dirs=test_list, config=config.data)
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=custom_collate_fn,
        num_workers=min(4, os.cpu_count()),  # 限制最大4个worker
        pin_memory=True
    )
    
    # 创建Trainer配置
    writer_callback = PredictionWriter(args.output_dir)
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        callbacks=[writer_callback],
        logger=False,
        enable_checkpointing=False,
        limit_predict_batches=config.predict.limit_batch_size
    )
    
    # 执行预测
    print(f"\n🚀 Starting inference on {len(dataset)} samples...")
    trainer.predict(model, dataloaders=dataloader)

if __name__ == "__main__":
    main()