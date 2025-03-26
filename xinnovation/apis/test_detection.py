import os
import torch
import argparse
from pathlib import Path
from xinnovation.src.core.config import Config
from xinnovation.src.core.registry import (
    BACKBONES, NECKS, HEADS, LOSSES,
    DATASETS, TRANSFORMS, SAMPLERS, LOADERS,
    METRICS, ANALYZERS, VISUALIZERS
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test a detection model')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--gpu-ids', type=int, nargs='+', help='ids of gpus to use')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', help='directory where painted images will be saved')
    parser.add_argument('--eval-options', nargs='+', help='custom options for evaluation')
    return parser.parse_args()

def build_model(cfg):
    """Build model from config."""
    # Build backbone
    backbone = BACKBONES.build(cfg.model.backbone)
    
    # Build neck
    neck = NECKS.build(cfg.model.neck)
    
    # Build head
    head = HEADS.build(cfg.model.head)
    
    # Build loss
    loss = LOSSES.build(cfg.model.loss)
    
    # Combine into model
    model = DetectionModel(
        backbone=backbone,
        neck=neck,
        head=head,
        loss=loss
    )
    
    return model

def build_dataset(cfg, default_args=None):
    """Build dataset from config."""
    if default_args is None:
        default_args = {}
        
    # Build transforms
    transforms = []
    for t in cfg.transforms:
        transforms.append(TRANSFORMS.build(t))
        
    # Build dataset
    dataset = DATASETS.build(
        cfg,
        default_args=dict(
            transforms=transforms,
            **default_args
        )
    )
    
    return dataset

def build_dataloader(dataset, cfg, default_args=None):
    """Build dataloader from config."""
    if default_args is None:
        default_args = {}
        
    # Build sampler
    sampler = SAMPLERS.build(
        cfg.sampler,
        default_args=dict(dataset=dataset, **default_args)
    )
    
    # Build dataloader
    dataloader = LOADERS.build(
        cfg,
        default_args=dict(
            dataset=dataset,
            sampler=sampler,
            **default_args
        )
    )
    
    return dataloader

def build_metrics(cfg):
    """Build metrics from config."""
    metrics = []
    for m in cfg:
        metrics.append(METRICS.build(m))
    return metrics

def build_analyzers(cfg):
    """Build analyzers from config."""
    analyzers = []
    for a in cfg:
        analyzers.append(ANALYZERS.build(a))
    return analyzers

def build_visualizers(cfg):
    """Build visualizers from config."""
    visualizers = []
    for v in cfg:
        visualizers.append(VISUALIZERS.build(v))
    return visualizers

def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load config
    cfg = Config(args.config)
    
    # Set work directory
    if args.work_dir is not None:
        cfg.test.work_dir = args.work_dir
    work_dir = Path(cfg.test.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Set GPU
    if args.gpu_ids is not None:
        cfg.test.gpu_ids = args.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build model
    model = build_model(cfg)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Build dataset and dataloader
    test_dataset = build_dataset(cfg.data.test)
    test_loader = build_dataloader(
        test_dataset,
        cfg.test,
        default_args=dict(num_workers=cfg.test.num_workers)
    )
    
    # Build evaluation components
    metrics = build_metrics(cfg.evaluation.metrics)
    analyzers = build_analyzers(cfg.evaluation.analyzers)
    visualizers = build_visualizers(cfg.evaluation.visualizers)
    
    # Create evaluator
    evaluator = DetectionEvaluator(
        model=model,
        test_loader=test_loader,
        metrics=metrics,
        analyzers=analyzers,
        visualizers=visualizers,
        work_dir=work_dir,
        device=device,
        show=args.show,
        show_dir=args.show_dir,
        eval_options=args.eval_options,
        **cfg.test
    )
    
    # Start evaluation
    evaluator.evaluate()

if __name__ == '__main__':
    main() 