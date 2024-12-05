import sys

import pandas as pd

sys.dont_write_bytecode = True

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
import os
from rich import print as rprint
import pytorch_lightning as pl
import random
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.datamodule import HandwritingDataModule
from src.models.RNN import RNN
from src.models.LSTM import LSTM
from src.utils.trainer_visualizer import TrainingVisualizer
from src.utils.print_info import check_cuda_availability, print_dataset_info, print_feature_info


def set_global_seed(seed: int) -> None:
    """Set seed for reproducibility across all libraries."""
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)
    

def configure_training(config):
    """
    Configure training with improved progress bar handling.
    
    This configuration provides a streamlined training setup that addresses progress bar
    truncation while using only supported PyTorch Lightning parameters. The setup includes
    GPU acceleration, early stopping, and a customized progress bar that shows essential
    metrics without overcrowding.
    
    Args:
        config: Configuration object containing training parameters
        
    Returns:
        dict: Trainer configuration dictionary for PyTorch Lightning
    """
    # Set GPU environment variables for consistent device selection
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Initialize the visualization callback for basic progress tracking
    visualizer = TrainingVisualizer()
    
    # Create progress bar callback with supported parameters
    progress_bar = pl.callbacks.progress.TQDMProgressBar(
        refresh_rate=1,  # Update every iteration
        process_position=0,  # Position in case of multiple progress bars
        leave=True  # Keep the progress bar after completion
    )
    
    # Create streamlined trainer configuration
    trainer_config = {
        # Hardware acceleration settings
        "accelerator": "gpu",
        "devices": 1,
        
        # Core training parameters
        "max_epochs": config.training.max_epochs,
        "gradient_clip_val": config.training.gradient_clip_val,
        
        # Essential callbacks for training management
        "callbacks": [
            EarlyStopping(
                monitor="val_mcc",
                patience=config.training.early_stopping_patience,
                mode="max",
                verbose=True
            ),
            progress_bar,
            visualizer
        ],
        
        # Progress bar settings
        "enable_progress_bar": True,
        "enable_model_summary": True,
        
        # Training settings
        "strategy": "auto",
        "sync_batchnorm": False,
        
        # Disable logging for cleaner output
        "logger": False
    }
    
    return trainer_config



@hydra.main(version_base="1.1", config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    try:
        set_global_seed(cfg.seed)
        rprint("[bold blue]Starting Handwriting Analysis with 5-Fold Cross Validation[/bold blue]")
        check_cuda_availability(cfg.verbose)
        
        fold_metrics = []
        
        for fold in range(5):
            rprint(f"\n[bold cyan]====== Starting Fold {fold + 1}/5 ======[/bold cyan]")
            
            data_module = HandwritingDataModule(
                data_dir=cfg.data.data_dir,
                batch_size=cfg.data.batch_size,
                window_size=cfg.data.window_size,
                stride=cfg.data.stride,
                num_workers=cfg.data.num_workers,
                num_tasks=cfg.data.num_tasks,
                file_pattern=cfg.data.file_pattern,
                column_names=dict(cfg.data.columns),
                fold=fold,
                n_folds=5,
                scaler_type=cfg.data.scaler,
                seed=cfg.seed,
                verbose=cfg.verbose
            )
            
            data_module.setup()
            
            print_feature_info(data_module) if cfg.verbose else None
            print_dataset_info(data_module) if cfg.verbose else None
            
            
            if cfg.model.type.lower() == "lstm":
                model = LSTM(
                    input_size=data_module.get_feature_dim(),
                    hidden_size=cfg.model.lstm_specific.hidden_size,
                    num_layers=cfg.model.lstm_specific.num_layers,
                    dropout=cfg.model.lstm_specific.dropout,
                    proj_size=cfg.model.lstm_specific.proj_size,
                    layer_norm=cfg.model.lstm_specific.layer_norm,
                    verbose=cfg.verbose
                )
            else:
                model = RNN(
                    input_size=data_module.get_feature_dim(),
                    hidden_size=cfg.model.rnn_specific.hidden_size,
                    num_layers=cfg.model.rnn_specific.num_layers,
                    verbose=cfg.verbose
                )
                
            model.model_config = {
                'learning_rate': cfg.training.learning_rate,
                'weight_decay': cfg.training.weight_decay
            }
            
            trainer_config = configure_training(cfg)
    
            trainer = pl.Trainer(**trainer_config)
            
            trainer.fit(model, data_module)
            
            fold_metrics.append({
                'fold': fold + 1,
                'val_loss': trainer.callback_metrics['val_loss'].item(),
                'val_acc': trainer.callback_metrics['val_acc'].item(),
                'val_f1': trainer.callback_metrics['val_f1'].item(),
                'val_mcc': trainer.callback_metrics['val_mcc'].item()
            })

            if cfg.verbose:
                rprint(f"\n[bold cyan]Fold {fold + 1}/5 completed![/bold cyan]")
                rprint(f"Validation Loss: {trainer.callback_metrics['val_loss']:.4f}")
                rprint(f"Validation Accuracy: {trainer.callback_metrics['val_acc']:.4f}")
                rprint(f"Validation F1 Score: {trainer.callback_metrics['val_f1']:.4f}")
                rprint(f"Validation MCC: {trainer.callback_metrics['val_mcc']:.4f}")
            
            if fold == 0:
                break
        
        metrics_df = pd.DataFrame(fold_metrics)
        rprint("\n[bold blue]Cross Validation Results:[/bold blue]")
        rprint(metrics_df.to_string())
        
        mean_metrics = metrics_df.mean()
        std_metrics = metrics_df.std()
        for metric in ['val_loss', 'val_acc', 'val_f1', 'val_mcc']:
            rprint(f"{metric}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")
            
    except Exception as e:
        rprint(f"[red]Error in main function: {str(e)}[/red]")
        raise e

if __name__ == "__main__":
    main()