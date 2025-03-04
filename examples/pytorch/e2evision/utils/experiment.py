import os
import logging
from typing import Optional, Union
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping
)
from lightning.pytorch.loggers import (
    TensorBoardLogger,
    WandbLogger,
    CSVLogger
)
from lightning.pytorch.strategies import DDPStrategy

from configs.config import Config
from models import E2EPerceptionModule
from data import E2EPerceptionDataModule

class ExperimentManager:
    """Manager for training experiments."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logging()
        
        # Set random seed
        L.seed_everything(config.seed)
        
        # Create components
        self.model = self._create_model()
        self.datamodule = self._create_datamodule()
        self.trainer = self._create_trainer()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup python logging."""
        logger = logging.getLogger("e2e_perception")
        logger.setLevel(logging.INFO)
        
        # Create handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(console_handler)
        
        # Add file handler if save_dir is specified
        if self.config.logging.save_dir:
            os.makedirs(self.config.logging.save_dir, exist_ok=True)
            file_handler = logging.FileHandler(
                os.path.join(self.config.logging.save_dir, 'experiment.log')
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _create_model(self) -> E2EPerceptionModule:
        """Create model from config."""
        return E2EPerceptionModule(
            camera_ids=self.config.data.camera_ids,
            feature_dim=self.config.model.feature_dim,
            num_queries=self.config.model.num_queries,
            num_decoder_layers=self.config.model.num_decoder_layers,
            hidden_dim=self.config.model.hidden_dim,
            dropout=self.config.model.dropout,
            num_attention_heads=self.config.model.num_attention_heads,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_epochs=self.config.training.warmup_epochs,
            max_epochs=self.config.training.max_epochs,
            lr_scheduler=self.config.training.lr_scheduler
        )
    
    def _create_datamodule(self) -> E2EPerceptionDataModule:
        """Create data module from config."""
        return E2EPerceptionDataModule(
            camera_ids=self.config.data.camera_ids,
            train_list=self.config.data.train_list,
            val_list=self.config.data.val_list,
            sequence_length=self.config.data.sequence_length,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            image_size=self.config.data.image_size,
            normalize_mean=self.config.data.normalize_mean,
            normalize_std=self.config.data.normalize_std
        )
    
    def _create_trainer(self) -> L.Trainer:
        """Create trainer from config."""
        # Create callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(self.config.logging.save_dir, 'checkpoints'),
            filename=f'{self.config.logging.experiment_name}-{{epoch:02d}}-{{val_loss:.2f}}',
            monitor=self.config.logging.monitor,
            mode=self.config.logging.monitor_mode,
            save_last=True,
            save_top_k=self.config.logging.save_top_k
        )
        callbacks.append(checkpoint_callback)
        
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=self.config.logging.monitor,
            mode=self.config.logging.monitor_mode,
            patience=10
        )
        callbacks.append(early_stopping)
        
        # Create loggers
        loggers = []
        
        # TensorBoard logger
        if self.config.logging.use_tensorboard:
            try:
                tb_logger = TensorBoardLogger(
                    save_dir=self.config.logging.save_dir,
                    name=self.config.logging.experiment_name,
                    default_hp_metric=False
                )
                loggers.append(tb_logger)
            except ModuleNotFoundError:
                self.logger.warning(
                    "TensorBoard not found. Please install with: pip install tensorboard"
                )
                csv_logger = CSVLogger(
                    save_dir=self.config.logging.save_dir,
                    name=self.config.logging.experiment_name
                )
                loggers.append(csv_logger)
        
        # Weights & Biases logger
        if self.config.logging.use_wandb:
            try:
                wandb_logger = WandbLogger(
                    project=self.config.logging.wandb_project,
                    entity=self.config.logging.wandb_entity,
                    name=self.config.logging.experiment_name
                )
                loggers.append(wandb_logger)
            except ModuleNotFoundError:
                self.logger.warning(
                    "Weights & Biases not found. Please install with: pip install wandb"
                )
        
        # Create training strategy
        if self.config.strategy == 'ddp':
            strategy = DDPStrategy(
                find_unused_parameters=False,
                gradient_as_bucket_view=True
            )
        else:
            strategy = self.config.strategy
        
        # Create trainer
        return L.Trainer(
            accelerator=self.config.accelerator,
            devices=self.config.devices,
            strategy=strategy,
            max_epochs=self.config.training.max_epochs,
            precision=self.config.training.precision,
            gradient_clip_val=self.config.training.gradient_clip_val,
            accumulate_grad_batches=self.config.training.accumulate_grad_batches,
            check_val_every_n_epoch=self.config.training.check_val_every_n_epoch,
            log_every_n_steps=self.config.training.log_every_n_steps,
            callbacks=callbacks,
            logger=loggers,
            deterministic=True
        )
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Train the model."""
        # Save config
        config_path = os.path.join(self.config.logging.save_dir, 'config.json')
        self.config.save(config_path)
        
        # Train model
        self.trainer.fit(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=resume_from_checkpoint
        )
    
    def test(self, checkpoint_path: Optional[str] = None):
        """Test the model."""
        self.trainer.test(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=checkpoint_path
        )
    
    def predict(self, checkpoint_path: str, output_dir: str):
        """Run prediction."""
        # Load checkpoint
        self.model = self.model.load_from_checkpoint(
            checkpoint_path,
            strict=True
        )
        
        # Create prediction trainer
        predict_trainer = L.Trainer(
            accelerator=self.config.accelerator,
            devices=1,
            precision=self.config.training.precision,
            logger=False
        )
        
        # Run prediction
        predictions = predict_trainer.predict(
            model=self.model,
            datamodule=self.datamodule
        )
        
        # Save predictions
        os.makedirs(output_dir, exist_ok=True)
        torch.save(predictions, os.path.join(output_dir, 'predictions.pt')) 