import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import wandb

class LogAnalyzer:
    """Analyze training logs and experiment results."""
    
    def __init__(self,
                 experiment_dir: Union[str, Path],
                 wandb_run: Optional[str] = None):
        self.exp_dir = Path(experiment_dir)
        self.wandb_run = wandb_run
        
        # Load metrics
        self.train_metrics = self._load_metrics('train')
        self.val_metrics = self._load_metrics('val')
        self.test_metrics = self._load_metrics('test')
        
        # Load git info if available
        self.git_info = self._load_git_info()
        
        # Load config
        self.config = self._load_config()
    
    def _load_metrics(self, phase: str) -> pd.DataFrame:
        """Load metrics from CSV file."""
        metrics_path = self.exp_dir / f'{phase}_metrics.csv'
        if metrics_path.exists():
            return pd.read_csv(metrics_path)
        return pd.DataFrame()
    
    def _load_git_info(self) -> Dict:
        """Load git repository information."""
        git_path = self.exp_dir / 'git_info.json'
        if git_path.exists():
            with open(git_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_config(self) -> Dict:
        """Load experiment configuration."""
        config_path = self.exp_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def plot_metrics(self,
                    metrics: List[str],
                    phases: List[str] = ['train', 'val'],
                    smoothing: float = 0.0,
                    save_path: Optional[str] = None) -> None:
        """Plot training metrics.
        
        Args:
            metrics: List of metric names to plot
            phases: List of phases to include
            smoothing: Exponential moving average factor
            save_path: Optional path to save plot
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4*n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            for phase in phases:
                df = getattr(self, f'{phase}_metrics')
                if metric in df.columns:
                    values = df[metric].values
                    steps = df['step'].values
                    
                    if smoothing > 0:
                        values = self._smooth_values(values, smoothing)
                    
                    ax.plot(steps, values, label=f'{phase}_{metric}')
            
            ax.set_xlabel('Step')
            ax.set_ylabel(metric)
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def analyze_convergence(self,
                          metric: str = 'loss',
                          window_size: int = 100) -> Dict[str, float]:
        """Analyze training convergence.
        
        Args:
            metric: Metric to analyze
            window_size: Window size for moving statistics
            
        Returns:
            Dict of convergence statistics
        """
        stats = {}
        
        for phase in ['train', 'val']:
            df = getattr(self, f'{phase}_metrics')
            if metric in df.columns:
                values = df[metric].values
                
                # Compute statistics
                stats[f'{phase}_final_{metric}'] = float(values[-1])
                stats[f'{phase}_best_{metric}'] = float(values.min())
                stats[f'{phase}_mean_{metric}'] = float(values.mean())
                stats[f'{phase}_std_{metric}'] = float(values.std())
                
                # Compute convergence indicators
                if len(values) > window_size:
                    recent_values = values[-window_size:]
                    recent_mean = recent_values.mean()
                    recent_std = recent_values.std()
                    
                    stats[f'{phase}_converged_{metric}'] = recent_std < 0.1 * recent_mean
                    stats[f'{phase}_convergence_rate'] = self._compute_convergence_rate(values)
        
        return stats
    
    def compare_experiments(self,
                          other_exp_dirs: List[Union[str, Path]],
                          metrics: List[str]) -> pd.DataFrame:
        """Compare metrics across multiple experiments.
        
        Args:
            other_exp_dirs: List of other experiment directories
            metrics: List of metrics to compare
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        # Add current experiment
        current_stats = self._get_experiment_stats(self.exp_dir, metrics)
        current_stats['experiment'] = self.exp_dir.name
        results.append(current_stats)
        
        # Add other experiments
        for exp_dir in other_exp_dirs:
            analyzer = LogAnalyzer(exp_dir)
            stats = analyzer._get_experiment_stats(exp_dir, metrics)
            stats['experiment'] = Path(exp_dir).name
            results.append(stats)
        
        return pd.DataFrame(results)
    
    def find_best_checkpoint(self,
                           metric: str = 'val_loss',
                           mode: str = 'min') -> str:
        """Find best checkpoint based on metric.
        
        Args:
            metric: Metric to use
            mode: 'min' or 'max'
            
        Returns:
            Path to best checkpoint
        """
        df = self.val_metrics
        if metric not in df.columns:
            raise ValueError(f"Metric {metric} not found in validation metrics")
        
        if mode == 'min':
            best_step = df.loc[df[metric].idxmin(), 'step']
        else:
            best_step = df.loc[df[metric].idxmax(), 'step']
        
        checkpoint_path = self.exp_dir / 'checkpoints' / f'checkpoint_{best_step:06d}.pth'
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def analyze_failure_cases(self,
                            threshold: float,
                            metric: str = 'loss') -> pd.DataFrame:
        """Analyze cases where model performed poorly.
        
        Args:
            threshold: Threshold for considering a case as failure
            metric: Metric to use
            
        Returns:
            DataFrame with failure cases
        """
        failures = []
        
        for phase in ['train', 'val', 'test']:
            df = getattr(self, f'{phase}_metrics')
            if metric in df.columns:
                # Find cases above threshold
                failure_cases = df[df[metric] > threshold]
                
                for _, case in failure_cases.iterrows():
                    failure_info = {
                        'phase': phase,
                        'step': case['step'],
                        metric: case[metric]
                    }
                    
                    # Add other metrics if available
                    for col in df.columns:
                        if col not in ['step', metric]:
                            failure_info[col] = case[col]
                    
                    failures.append(failure_info)
        
        return pd.DataFrame(failures)
    
    def export_analysis(self, output_path: Union[str, Path]):
        """Export analysis results to HTML report.
        
        Args:
            output_path: Path to save HTML report
        """
        import jinja2
        
        # Prepare data for report
        data = {
            'experiment_name': self.exp_dir.name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'git_info': self.git_info,
            'config': self.config,
            'convergence_stats': self.analyze_convergence(),
            'train_metrics': self.train_metrics.describe().to_html(),
            'val_metrics': self.val_metrics.describe().to_html()
        }
        
        # Create plots
        plot_dir = Path(output_path).parent / 'plots'
        plot_dir.mkdir(exist_ok=True)
        
        # Loss plot
        self.plot_metrics(
            metrics=['loss'],
            save_path=str(plot_dir / 'loss.png')
        )
        data['loss_plot'] = 'plots/loss.png'
        
        # Metrics plot
        metric_names = [col for col in self.train_metrics.columns
                       if col not in ['step', 'loss']]
        if metric_names:
            self.plot_metrics(
                metrics=metric_names,
                save_path=str(plot_dir / 'metrics.png')
            )
            data['metrics_plot'] = 'plots/metrics.png'
        
        # Load template
        template = """
        <html>
        <head>
            <title>Experiment Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; margin: 10px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; }
                img { max-width: 100%; }
            </style>
        </head>
        <body>
            <h1>Experiment Analysis Report</h1>
            <p>Generated on: {{ timestamp }}</p>
            
            <h2>Experiment Information</h2>
            <p>Name: {{ experiment_name }}</p>
            
            {% if git_info %}
            <h3>Git Information</h3>
            <ul>
                <li>Commit: {{ git_info.commit_hash }}</li>
                <li>Branch: {{ git_info.branch }}</li>
                <li>Status: {% if git_info.is_dirty %}Dirty{% else %}Clean{% endif %}</li>
            </ul>
            {% endif %}
            
            <h2>Configuration</h2>
            <pre>{{ config | tojson(indent=2) }}</pre>
            
            <h2>Convergence Statistics</h2>
            <pre>{{ convergence_stats | tojson(indent=2) }}</pre>
            
            <h2>Training Metrics</h2>
            {{ train_metrics }}
            
            <h2>Validation Metrics</h2>
            {{ val_metrics }}
            
            <h2>Plots</h2>
            {% if loss_plot %}
            <h3>Loss</h3>
            <img src="{{ loss_plot }}">
            {% endif %}
            
            {% if metrics_plot %}
            <h3>Other Metrics</h3>
            <img src="{{ metrics_plot }}">
            {% endif %}
        </body>
        </html>
        """
        
        # Render template
        html = jinja2.Template(template).render(**data)
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html)
    
    @staticmethod
    def _smooth_values(values: np.ndarray, alpha: float) -> np.ndarray:
        """Apply exponential moving average smoothing."""
        smoothed = np.zeros_like(values)
        smoothed[0] = values[0]
        for i in range(1, len(values)):
            smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    @staticmethod
    def _compute_convergence_rate(values: np.ndarray,
                                window_size: int = 100) -> float:
        """Compute convergence rate using linear regression."""
        if len(values) < window_size:
            return 0.0
        
        recent_values = values[-window_size:]
        x = np.arange(window_size)
        y = recent_values
        
        # Fit line to recent values
        coeffs = np.polyfit(x, y, deg=1)
        return abs(coeffs[0])  # Return absolute slope
    
    @staticmethod
    def _get_experiment_stats(exp_dir: Union[str, Path],
                            metrics: List[str]) -> Dict:
        """Get summary statistics for an experiment."""
        analyzer = LogAnalyzer(exp_dir)
        stats = {}
        
        for phase in ['train', 'val']:
            df = getattr(analyzer, f'{phase}_metrics')
            for metric in metrics:
                if metric in df.columns:
                    values = df[metric].values
                    stats[f'{phase}_{metric}_final'] = float(values[-1])
                    stats[f'{phase}_{metric}_best'] = float(values.min())
                    stats[f'{phase}_{metric}_mean'] = float(values.mean())
                    stats[f'{phase}_{metric}_std'] = float(values.std())
        
        return stats 