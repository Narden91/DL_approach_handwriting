from typing import Any, Optional
from rich import print as rprint
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig


def configure_wandb_logger(config: DictConfig, fold: int, window_size: int, stride: int) -> Optional[WandbLogger]:
    """Configure WandB logger with proper cleanup between folds.
    
    Args:
        config: Configuration object containing experiment settings
        fold: Current fold number
        window_size: Current window size
        stride: Current stride value
        
    Returns:
        WandbLogger instance or None if initialization fails
    """
    try:
        # Import wandb and finish any existing runs
        import wandb
        if wandb.run is not None:
            wandb.finish()
            
        # Initialize new wandb logger
        wandb_logger = WandbLogger(
            project="handwriting_analysis",
            name=f"{config.model.type}_fold{fold+1}_ws{window_size}_str{stride}",
            group=f"{config.experiment_name}",
            tags=[f"fold_{fold+1}", f"window_size_{window_size}", f"stride_{stride}"]
        )
        
        # Log hyperparameters
        wandb_logger.log_hyperparams({
            "model_type": config.model.type,
            "fold": fold + 1,
            "window_size": window_size,
            "stride": stride,
            "batch_size": config.data.batch_size,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay
        })
        
        return wandb_logger
        
    except Exception as e:
        rprint(f"[yellow]Warning: Could not initialize WandB logger: {str(e)}. Continuing without logging...[/yellow]")
        return None

def cleanup_wandb() -> None:
    """Cleanup WandB run after training completion."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception as e:
        rprint(f"[yellow]Warning: Error cleaning up WandB: {str(e)}[/yellow]")