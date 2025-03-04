import os
import json
import shutil
from typing import Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
import wandb
from pathlib import Path
import git

class ExperimentTracker:
    """Track and manage experiment results and artifacts."""
    
    def __init__(self, 
                 experiment_name: str,
                 base_dir: str = 'experiments',
                 use_wandb: bool = False,
                 wandb_project: Optional[str] = None,
                 wandb_entity: Optional[str] = None):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.use_wandb = use_wandb
        
        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Initialize wandb if requested
        self.wandb_run = None
        if use_wandb:
            self.wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=f"{experiment_name}_{timestamp}",
                config={},
                dir=self.exp_dir
            )
        
        # Initialize metrics storage
        self.metrics = {
            'train': [],
            'val': [],
            'test': []
        }
        
        # Save git info if available
        self._save_git_info()
    
    def _save_git_info(self):
        """Save git repository information."""
        try:
            repo = git.Repo(search_parent_directories=True)
            git_info = {
                'commit_hash': repo.head.object.hexsha,
                'branch': repo.active_branch.name,
                'is_dirty': repo.is_dirty(),
                'modified_files': [item.a_path for item in repo.index.diff(None)],
                'untracked_files': repo.untracked_files
            }
            
            with open(os.path.join(self.exp_dir, 'git_info.json'), 'w') as f:
                json.dump(git_info, f, indent=2)
                
        except git.InvalidGitRepositoryError:
            print("Warning: Not a git repository")
    
    def save_config(self, config: Dict):
        """Save experiment configuration."""
        config_path = os.path.join(self.exp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        if self.wandb_run is not None:
            self.wandb_run.config.update(config)
    
    def log_metrics(self, 
                   metrics: Dict[str, float],
                   step: int,
                   phase: str = 'train'):
        """Log training metrics."""
        # Add step number
        metrics['step'] = step
        
        # Save to local storage
        self.metrics[phase].append(metrics)
        
        # Save to wandb if enabled
        if self.wandb_run is not None:
            self.wandb_run.log({f"{phase}/{k}": v for k, v in metrics.items()})
        
        # Save to CSV
        df = pd.DataFrame(self.metrics[phase])
        df.to_csv(os.path.join(self.exp_dir, f'{phase}_metrics.csv'), index=False)
    
    def save_model(self, 
                  state_dict: Dict,
                  step: int,
                  is_best: bool = False):
        """Save model checkpoint."""
        # Save checkpoint
        checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'checkpoint_{step:06d}.pth'
        )
        torch.save(state_dict, checkpoint_path)
        
        # Save as best model if specified
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            shutil.copyfile(checkpoint_path, best_path)
        
        # Save to wandb if enabled
        if self.wandb_run is not None:
            artifact = wandb.Artifact(
                name=f"{self.experiment_name}_model",
                type='model',
                metadata={'step': step}
            )
            artifact.add_file(checkpoint_path)
            self.wandb_run.log_artifact(artifact)
    
    def save_predictions(self,
                        predictions: Dict,
                        step: int):
        """Save model predictions."""
        # Save predictions
        pred_dir = os.path.join(self.exp_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        
        pred_path = os.path.join(pred_dir, f'predictions_{step:06d}.pth')
        torch.save(predictions, pred_path)
        
        # Save to wandb if enabled
        if self.wandb_run is not None:
            artifact = wandb.Artifact(
                name=f"{self.experiment_name}_predictions",
                type='predictions',
                metadata={'step': step}
            )
            artifact.add_file(pred_path)
            self.wandb_run.log_artifact(artifact)
    
    def save_visualizations(self,
                          visualizations: Dict[str, wandb.Image],
                          step: int):
        """Save visualizations."""
        # Save locally
        vis_dir = os.path.join(self.exp_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        for name, img in visualizations.items():
            img_path = os.path.join(vis_dir, f'{name}_{step:06d}.png')
            img.save(img_path)
        
        # Log to wandb if enabled
        if self.wandb_run is not None:
            self.wandb_run.log(
                {f"vis/{k}": v for k, v in visualizations.items()},
                step=step
            )
    
    def save_profiling_results(self,
                             results: Dict[str, Dict[str, float]]):
        """Save profiling results."""
        # Save as JSON
        profile_path = os.path.join(self.exp_dir, 'profiling_results.json')
        with open(profile_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Log to wandb if enabled
        if self.wandb_run is not None:
            self.wandb_run.summary.update(
                {f"profile/{k1}/{k2}": v2 
                 for k1, v1 in results.items()
                 for k2, v2 in v1.items()}
            )
    
    def finish(self):
        """Finish experiment tracking."""
        if self.wandb_run is not None:
            self.wandb_run.finish()
    
    def load_checkpoint(self, step: Optional[int] = None) -> Dict:
        """Load model checkpoint.
        
        Args:
            step: Specific step to load, or None for latest
            
        Returns:
            Model state dict
        """
        checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        
        if step is not None:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'checkpoint_{step:06d}.pth'
            )
        else:
            # Find latest checkpoint
            checkpoints = sorted(Path(checkpoint_dir).glob('checkpoint_*.pth'))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found")
            checkpoint_path = str(checkpoints[-1])
        
        return torch.load(checkpoint_path)
    
    def get_metrics_summary(self, phase: str = 'train') -> pd.DataFrame:
        """Get summary of tracked metrics.
        
        Args:
            phase: One of ['train', 'val', 'test']
            
        Returns:
            DataFrame containing metrics
        """
        return pd.DataFrame(self.metrics[phase]) 