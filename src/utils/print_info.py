from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from rich.box import ROUNDED
import torch   
import pandas as pd
from src.data.datamodule import HandwritingDataModule


def check_cuda_availability(verbose: bool = True):
    """Check CUDA availability and print detailed information."""
    cuda_available = torch.cuda.is_available()
    
    if verbose:
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
    splits_table = Table(title="Dataset Splits", show_header=True, header_style="bold magenta", box=ROUNDED)
    splits_table.add_column("Split", style="cyan")
    splits_table.add_column("Size", style="green")
    splits_table.add_column("Windows", style="yellow")
    
    # Add rows for each split - using index for unique subjects
    splits_table.add_row(
        "Training",
        str(len(data_module.train_dataset.data.index.get_level_values(0).unique())),
        str(len(data_module.train_dataset))
    )
    splits_table.add_row(
        "Validation",
        str(len(data_module.val_dataset.data.index.get_level_values(0).unique())),
        str(len(data_module.val_dataset))
    )
    splits_table.add_row(
        "Test",
        str(len(data_module.test_dataset.data.index.get_level_values(0).unique())),
        str(len(data_module.test_dataset))
    )
    
    # Create a table for feature information
    feature_table = Table(title="Feature Information", show_header=True, header_style="bold magenta", box=ROUNDED)
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
    
    class_table = Table(title="Class Distribution (Training Set)", show_header=True, header_style="bold magenta", box=ROUNDED)
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
    
    # Print dataframe heads
    rprint("\n[bold blue]Dataset Previews[/bold blue]")
    print_dataframe_head(data_module.train_dataset.data, "[bold green]Training Dataset Head[/bold green]")
    print_dataframe_head(data_module.val_dataset.data, "[bold yellow]Validation Dataset Head[/bold yellow]")
    print_dataframe_head(data_module.test_dataset.data, "[bold red]Test Dataset Head[/bold red]")


def print_dataframe_head(df: pd.DataFrame, title: str, n_rows: int = 5):
    """Print the head of a dataframe using rich table formatting."""
    table = Table(title=title, box=ROUNDED, show_header=True, header_style="bold magenta")
    
    # Add index columns first
    index_names = df.index.names
    for idx_name in index_names:
        table.add_column(str(idx_name), style="cyan")
    
    # Add data columns
    for column in df.columns:
        table.add_column(str(column), style="cyan")
    
    # Add rows
    for idx, row in df.head(n_rows).iterrows():
        # Handle multi-index
        if isinstance(idx, tuple):
            idx_values = [str(x) for x in idx]
        else:
            idx_values = [str(idx)]
        
        # Convert row values to formatted strings
        row_values = []
        for value in row:
            if isinstance(value, float):
                row_values.append(f"{value:.4f}")
            else:
                row_values.append(str(value))
        
        # Combine index and row values
        table.add_row(*(idx_values + row_values))
    
    rprint(table)
    
    
def print_feature_info(data_module: HandwritingDataModule):
    """Print detailed information about input features using rich formatting."""
    feature_table = Table(
        title="Input Feature Information",
        show_header=True,
        header_style="bold magenta",
        box=ROUNDED
    )
    
    feature_table.add_column("Category", style="cyan")
    feature_table.add_column("Value", style="green")
    
    feature_table.add_row(
        "Feature Dimension",
        str(data_module.get_feature_dim())
    )
    
    feature_list_table = Table(
        title="Feature List",
        show_header=True,
        header_style="bold magenta",
        box=ROUNDED
    )
    
    feature_list_table.add_column("Index", style="cyan")
    feature_list_table.add_column("Feature Name", style="green")
    
    for idx, feature in enumerate(data_module.feature_cols):
        feature_list_table.add_row(str(idx), feature)
    
    rprint("\n[bold blue]Input Feature Details[/bold blue]")
    rprint(feature_table)
    rprint(feature_list_table)
    

def print_sets_info(data_module: HandwritingDataModule, fold: int):
    """Print detailed information about dataset splits using rich formatting."""
    splits_table = Table(
        title="Dataset Splits",
        show_header=True,
        header_style="bold magenta",
        box=ROUNDED
    )
    
    splits_table.add_column("Split", style="cyan")
    splits_table.add_column("Size", style="green")
    splits_table.add_column("Subjects", style="yellow")
    
    train_subjects = data_module.train_dataset.data.index.get_level_values(0).unique()
    val_subjects = data_module.val_dataset.data.index.get_level_values(0).unique()
    test_subjects = data_module.test_dataset.data.index.get_level_values(0).unique()
    
    splits_table.add_row("Training", str(len(data_module.train_dataset)), str(len(train_subjects)))
    splits_table.add_row("Validation", str(len(data_module.val_dataset)), str(len(val_subjects)))
    splits_table.add_row("Test", str(len(data_module.test_dataset)), str(len(test_subjects)))
    
    rprint("\n[bold blue]Dataset Split Information[/bold blue]")
    rprint(splits_table)
    

def print_predictions(subjects, labels, preds, fold, set_type):
    """Print predictions for a given set (train/test)."""
    rprint(f"\n[bold blue]{set_type.capitalize()} Predictions for Fold {fold + 1}:[/bold blue]")
    for subject, label, pred in zip(subjects, labels, preds):
        rprint(f"Subject: {subject}, True Label: {label}, Predicted Label: {pred}")


def print_fold_completion(fold, trainer):
    """Print fold completion details."""
    rprint(f"\n[bold cyan]Fold {fold + 1}/5 completed![/bold cyan]")
    rprint(f"Validation Loss: {trainer.callback_metrics['val_loss']:.4f}")
    rprint(f"Validation Accuracy: {trainer.callback_metrics['val_acc']:.4f}")
    rprint(f"Validation F1 Score: {trainer.callback_metrics['val_f1']:.4f}")
    rprint(f"Validation MCC: {trainer.callback_metrics['val_mcc']:.4f}")