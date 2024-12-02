import sys

sys.dont_write_bytecode = True

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
from pathlib import Path
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

from src.data.datamodule import HandwritingDataModule
from src.utils.print_info import check_cuda_availability, print_dataset_info


@hydra.main(version_base="1.1", config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    try:
        rprint("[bold blue]Starting Handwriting Analysis Experiment[/bold blue]")
        
        # Check CUDA availability
        check_cuda_availability()
        
        # Print configuration
        rprint("\n[yellow]Configuration:[/yellow]")
        rprint(f"Experiment Name: {cfg.experiment_name}")
        rprint(f"Random Seed: {cfg.seed}")
        rprint(f"Device: {cfg.device}")
        
        rprint("\n[yellow]Data Configuration:[/yellow]")
        rprint(f"Data Directory: {cfg.data.data_dir}")
        rprint(f"Window Size: {cfg.data.window_size}")
        rprint(f"Stride: {cfg.data.stride}")
        rprint(f"Batch Size: {cfg.data.batch_size}")
        rprint(f"Num Workers: {cfg.data.num_workers}")
        rprint(f"Validation Split: {cfg.data.val_split}")
        rprint(f"Test Split: {cfg.data.test_split}")
        
        # Set random seeds
        pl.seed_everything(cfg.seed, workers=True)
        
        rprint("\n[green]Initializing data module...[/green]")
        
        # Initialize data module
        data_module = HandwritingDataModule(
            data_dir=cfg.data.data_dir,
            batch_size=cfg.data.batch_size,
            window_size=cfg.data.window_size,
            stride=cfg.data.stride,
            num_workers=cfg.data.num_workers,
            val_split=cfg.data.val_split,
            test_split=cfg.data.test_split,
            num_tasks=cfg.data.num_tasks,
            file_pattern=cfg.data.file_pattern,
            column_names=dict(cfg.data.columns),
            seed=cfg.seed
        )
        
        # Setup data module
        data_module.setup()
        
        # Print detailed dataset information
        print_dataset_info(data_module)
        
        # Get example batch information
        train_loader = data_module.train_dataloader()
        features, labels, task_ids, masks = next(iter(train_loader))
        
        rprint("\n[yellow]Example Batch Information:[/yellow]")
        rprint(f"Features shape: {features.shape}")
        rprint(f"Labels shape: {labels.shape}")
        rprint(f"Task IDs shape: {task_ids.shape}")
        rprint(f"Masks shape: {masks.shape}")
        
        rprint("\n[blue]Data Module Initialization Complete[/blue]")
            
    except Exception as e:
        rprint(f"[red]Error in main function: {str(e)}[/red]")
        raise e

if __name__ == "__main__":
    main()

