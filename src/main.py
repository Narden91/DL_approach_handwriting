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

from data.datamodule import HandwritingDataModule


def check_cuda_availability():
    """Check CUDA availability and print detailed information."""
    cuda_available = torch.cuda.is_available()
    
    rprint(Panel.fit(
        f"[bold blue]CUDA Information:[/bold blue]\n"
        f"CUDA Available: [green]{cuda_available}[/green]\n"
        f"PyTorch Version: [yellow]{torch.__version__}[/yellow]"
    ))
    
    if cuda_available:
        rprint(f"CUDA Version: [yellow]{torch.version.cuda}[/yellow]")
        rprint(f"Current Device: [yellow]{torch.cuda.current_device()}[/yellow]")
        rprint(f"Device Name: [yellow]{torch.cuda.get_device_name(0)}[/yellow]")
        rprint(f"Device Count: [yellow]{torch.cuda.device_count()}[/yellow]")
        rprint(f"Memory Allocated: [yellow]{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB[/yellow]")
        rprint(f"Memory Cached: [yellow]{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB[/yellow]")

def print_dataset_info(data_module: HandwritingDataModule):
    """Print detailed dataset information using rich formatting."""
    # Create a table for dataset splits
    splits_table = Table(title="Dataset Splits", show_header=True, header_style="bold magenta")
    splits_table.add_column("Split", style="cyan")
    splits_table.add_column("Size", style="green")
    splits_table.add_column("Windows", style="yellow")
    
    # Add rows for each split
    splits_table.add_row(
        "Training",
        str(len(data_module.train_dataset.data[data_module.column_names['id']].unique())),
        str(len(data_module.train_dataset))
    )
    splits_table.add_row(
        "Validation",
        str(len(data_module.val_dataset.data[data_module.column_names['id']].unique())),
        str(len(data_module.val_dataset))
    )
    splits_table.add_row(
        "Test",
        str(len(data_module.test_dataset.data[data_module.column_names['id']].unique())),
        str(len(data_module.test_dataset))
    )
    
    # Create a table for feature information
    feature_table = Table(title="Feature Information", show_header=True, header_style="bold magenta")
    feature_table.add_column("Category", style="cyan")
    feature_table.add_column("Details", style="green")
    
    feature_table.add_row("Number of Features", str(data_module.get_feature_dim()))
    feature_table.add_row("Number of Tasks", str(data_module.get_num_tasks()))
    feature_table.add_row("Window Size", str(data_module.window_size))
    feature_table.add_row("Stride", str(data_module.stride))
    feature_table.add_row("Batch Size", str(data_module.batch_size))
    
    # Print class distribution
    label_col = data_module.column_names['label']
    class_dist = data_module.train_dataset.data[label_col].value_counts()
    
    class_table = Table(title="Class Distribution (Training Set)", show_header=True, header_style="bold magenta")
    class_table.add_column("Class", style="cyan")
    class_table.add_column("Count", style="green")
    class_table.add_column("Percentage", style="yellow")
    
    total = class_dist.sum()
    for label, count in class_dist.items():
        percentage = (count / total) * 100
        class_table.add_row(
            str(label),
            str(count),
            f"{percentage:.2f}%"
        )
    
    # Print all tables
    rprint("\n[bold blue]Dataset Information[/bold blue]")
    rprint(splits_table)
    rprint(feature_table)
    rprint(class_table)

@hydra.main(version_base="1.1", config_path="../conf", config_name="config")
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

