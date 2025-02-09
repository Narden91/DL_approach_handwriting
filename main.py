from datetime import datetime
import sys
sys.dont_write_bytecode = True
from src.utils.wandb_utils import cleanup_wandb
import pandas as pd
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
import os
from rich import print as rprint
import random
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from src.utils.model_factory import ModelFactory
from src.data.datamodule import DataConfig, HandwritingDataModule
from src.utils.callbacks import GradientMonitorCallback, ThresholdTuner
from s3_operations.s3_handler import config
from s3_operations.s3_io import S3IOHandler
from src.utils.config_operations import ConfigOperations
from src.utils.print_info import check_cuda_availability, print_dataset_info, print_feature_info, print_subject_metrics, \
    process_metrics
from src.utils.majority_vote import get_predictions, compute_subject_metrics
from src.explainability.model_explainer import GradientModelExplainer


def set_global_seed(seed: int) -> None:
    """Set seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)


def save_importance_data(s3_handler, importance_data, file_key_save, importance_type="feature"):
    """
    Save importance data to CSV via S3 using pre-defined file paths
    
    Args:
        s3_handler: S3IOHandler instance
        importance_data: Dictionary containing importance values
        file_key_save: Complete S3 file path for saving
        importance_type: Type of importance data ("feature" or "task")
    """
    try:
        # Create DataFrame from importance dictionary
        if importance_type == "feature":
            df = pd.DataFrame({
                'feature': list(importance_data.keys()),
                'importance': list(importance_data.values())
            })
        else:  # task importance
            df = pd.DataFrame({
                'task_id': list(importance_data.keys()),
                'importance': list(importance_data.values())
            })
        
        # Sort by importance value descending
        df = df.sort_values('importance', ascending=False)
        
        # Save to S3
        s3_handler.save_data(df, file_key_save)
        rprint(f"[green]Successfully saved {importance_type} importance results to {file_key_save}[/green]")
        
    except Exception as e:
        rprint(f"[red]Error saving {importance_type} importance data: {str(e)}[/red]")
        

def configure_training(config):
    """Configure training with wandb logger and callbacks."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Initialize callbacks
    progress_bar = pl.callbacks.progress.TQDMProgressBar(
        refresh_rate=1,
        process_position=0,
    )

    early_stopping = EarlyStopping(
        monitor="val_f1",
        patience=config.training.early_stopping_patience,
        mode="max",
        verbose=True
    )

    threshold_tuner = ThresholdTuner()

    try:
        # Initialize Wandb logger with error handling
        wandb_logger = WandbLogger(
            project="handwriting_analysis",
            name=f"{config.model.type}_ws{config.data.window_sizes}_str{config.data.strides}",
        )
    except Exception as e:
        rprint(f"[yellow]Warning: Could not initialize WandB logger: {str(e)}. Continuing without logging...[/yellow]")
        wandb_logger = None

    trainer_config = {
        "accelerator": "gpu",
        "devices": 1,
        "max_epochs": config.training.max_epochs,
        "gradient_clip_val": config.training.gradient_clip_val,
        "callbacks": [
            early_stopping,
            progress_bar,
            threshold_tuner,
            GradientMonitorCallback()
        ],
        "logger": wandb_logger if wandb_logger is not None else False,
        "log_every_n_steps": 10,
        "val_check_interval": 0.5,
        "enable_checkpointing": False,
        "deterministic": True
    }

    return trainer_config, wandb_logger


def safe_wandb_log(logger, metrics, epoch=None):
    """Safely log metrics to WandB with error handling."""
    if logger is not None:
        try:
            if epoch is not None:
                metrics['epoch'] = epoch
            logger.log_metrics(metrics)
        except Exception as e:
            rprint(f"[yellow]Warning: Failed to log metrics to WandB: {str(e)}[/yellow]")


@hydra.main(version_base="1.1", config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    try:
        # Initialize configuration and set seeds
        cfg = ConfigOperations.merge_configurations(cfg)
        set_global_seed(cfg.seed)
        rprint("[bold blue]Starting Handwriting Analysis with 5-Fold Cross Validation[/bold blue]")
        check_cuda_availability(cfg.verbose)

        # Setup file paths
        file_key_load: str = f"{cfg.data.s3_folder_input}/{cfg.data.data_filename}"
        result_output_filename: str = f"{cfg.data.output_filename}_{cfg.model.type}_DA_{cfg.data.enable_augmentation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        feature_output_filename: str = f"Feature_importance_{cfg.model.type}_DA_{cfg.data.enable_augmentation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        task_output_filename: str = f"Task_importance_{cfg.model.type}_DA_{cfg.data.enable_augmentation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_key_save: str = f"{cfg.data.s3_folder_output}/{result_output_filename}"
        feature_key_save: str = f"{cfg.data.s3_folder_output}/{feature_output_filename}"
        task_key_save: str = f"{cfg.data.s3_folder_output}/{task_output_filename}"
        
        # Initialize S3 handler
        s3_handler: S3IOHandler = S3IOHandler(config, verbose=cfg.verbose)
        
        yaml_split_path = cfg.data.get('yaml_split_path', None) 

        # Check if yaml split path exists in local directory
        if yaml_split_path is not None:
            if not os.path.exists(yaml_split_path):
                rprint(f"[red]Error: YAML split file not found at {yaml_split_path}. Exiting...[/red]")
                return

        # Initialize metrics lists
        fold_metrics = []
        feature_importance_all = []
        task_importance_all = []

        # Main loop over window sizes and strides
        for window_size in cfg.data.window_sizes:
            for stride in cfg.data.strides:
                # Initialize lists for this window/stride configuration
                feature_importance_folds = []
                task_importance_folds = []
                
                rprint(f"\n[bold cyan]====== Testing window_size={window_size}, stride={stride} ======[/bold cyan]")
                
                # Fold loop
                for fold in range(cfg.num_folds):
                    rprint(f"\n[bold cyan]====== Starting Fold {fold + 1}/5 ======[/bold cyan]")
                    try:
                        # Set fold-specific seed
                        fold_seed = cfg.seed + fold

                        # Before creating the HandwritingDataModule, create a DataConfig instance
                        data_config = DataConfig(
                            window_size=window_size,
                            stride=stride,
                            batch_size=cfg.data.batch_size,
                            num_workers=cfg.data.num_workers,
                            scaler_type=cfg.data.scaler,
                            verbose=cfg.verbose
                        )

                        # Now create the DataModule with the config object
                        data_module = HandwritingDataModule(
                            s3_handler=s3_handler,
                            file_key=file_key_load,
                            config=data_config,  
                            column_names=dict(cfg.data.columns),
                            fold=fold,
                            n_folds=cfg.num_folds,
                            seed=fold_seed,
                            yaml_split_path=yaml_split_path
                        )

                        # Setup data
                        data_module.setup()

                        # Print info if verbose
                        if cfg.verbose:
                            print_feature_info(data_module)
                            print_dataset_info(data_module)
                        
                        # Create and configure model
                        model = ModelFactory.create_model(cfg, data_module)
                        model.model_config = {
                            'learning_rate': cfg.training.learning_rate,
                            'weight_decay': cfg.training.weight_decay
                        }

                        # Set class weights if available
                        if hasattr(data_module.train_dataset, 'class_weights'):
                            model.set_class_weights(data_module.train_dataset)

                        # Configure training and fit model
                        trainer_config, wandb_logger = configure_training(cfg)
                        trainer = pl.Trainer(**trainer_config)
                        trainer.fit(model, data_module)
                        
                        # Initialize explainer
                        explainer = GradientModelExplainer(
                            model=model,
                            datamodule=data_module,
                            feature_names=data_module.feature_cols,
                            verbose=cfg.verbose,
                            n_samples=100,
                            batch_size=cfg.data.batch_size
                        )

                        # Calculate feature and task importance
                        feature_importance = explainer.analyze_feature_importance(fold)
                        task_importance = explainer.analyze_task_importance(fold)

                        # Store importance dictionaries for aggregation
                        feature_importance_folds.append(feature_importance)
                        task_importance_folds.append(task_importance)

                        # Get predictions
                        train_subjects, train_labels, train_preds = get_predictions(
                            trainer, model, data_module.train_dataloader()
                        )
                        test_subjects, test_labels, test_preds = get_predictions(
                            trainer, model, data_module.test_dataloader()
                        )

                        # Compute metrics
                        train_subject_metrics = compute_subject_metrics(
                            train_subjects, train_labels, train_preds, cfg.verbose
                        )
                        test_subject_metrics = compute_subject_metrics(
                            test_subjects, test_labels, test_preds, cfg.verbose
                        )

                        # Store metrics for this fold
                        metrics = {
                            'window_size': window_size,
                            'stride': stride,
                            'fold': fold + 1,
                            'train_subject_acc': train_subject_metrics['subject_accuracy'],
                            'train_subject_precision': train_subject_metrics['subject_precision'],
                            'train_subject_recall': train_subject_metrics['subject_recall'],
                            'train_subject_specificity': train_subject_metrics['subject_specificity'],
                            'train_subject_f1': train_subject_metrics['subject_f1'],
                            'train_subject_mcc': train_subject_metrics['subject_mcc'],
                            'test_subject_acc': test_subject_metrics['subject_accuracy'],
                            'test_subject_precision': test_subject_metrics['subject_precision'],
                            'test_subject_recall': test_subject_metrics['subject_recall'],
                            'test_subject_specificity': test_subject_metrics['subject_specificity'],
                            'test_subject_f1': test_subject_metrics['subject_f1'],
                            'test_subject_mcc': test_subject_metrics['subject_mcc'],
                        }

                        # Log metrics to wandb
                        safe_wandb_log(wandb_logger, metrics, trainer.current_epoch)

                        # Store metrics
                        fold_metrics.append(metrics)
                        print_subject_metrics(train_subject_metrics, test_subject_metrics, fold, cfg.verbose)

                        cleanup_wandb()
                        
                        if cfg.test_mode and fold == 0:
                            break

                    except Exception as e:
                        rprint(f"[red]Error in fold {fold + 1}: {str(e)}[/red]")
                        return e

                # After all folds for this configuration, aggregate the importance results
                avg_feature_importance = GradientModelExplainer.aggregate_importances(feature_importance_folds)
                avg_task_importance = GradientModelExplainer.aggregate_importances(task_importance_folds)

                # Store aggregated results with configuration metadata
                for feature, importance in avg_feature_importance.items():
                    feature_importance_all.append({
                        'window_size': window_size,
                        'stride': stride,
                        'feature': feature,
                        'importance': importance
                    })

                for task_id, importance in avg_task_importance.items():
                    task_importance_all.append({
                        'window_size': window_size,
                        'stride': stride,
                        'task_id': task_id,
                        'importance': importance
                    })

        # After all configurations, save results
        if fold_metrics:
            # Save metrics
            metrics_df = pd.DataFrame(fold_metrics)
            process_metrics(metrics_df, window_size, stride, cfg)
            
            try:
                # Save regular metrics
                s3_handler.save_data(metrics_df, file_key_save)
                rprint(f"[green]Successfully saved metrics to {file_key_save}[/green]")

                # Save feature importance
                if feature_importance_all:
                    feature_importance_df = pd.DataFrame(feature_importance_all)
                    s3_handler.save_data(feature_importance_df, feature_key_save)
                    rprint(f"[green]Successfully saved feature importance to {feature_key_save}[/green]")

                # Save task importance
                if task_importance_all:
                    task_importance_df = pd.DataFrame(task_importance_all)
                    s3_handler.save_data(task_importance_df, task_key_save)
                    rprint(f"[green]Successfully saved task importance to {task_key_save}[/green]")

                # Print results if verbose
                if cfg.verbose:
                    # Print feature importance for each configuration
                    for ws in cfg.data.window_sizes:
                        for st in cfg.data.strides:
                            config_features = [
                                row for row in feature_importance_all 
                                if row['window_size'] == ws and row['stride'] == st
                            ]
                            if config_features:
                                rprint(f"\n[bold blue]Feature Importance for window_size={ws}, stride={st}:[/bold blue]")
                                for row in sorted(config_features, key=lambda x: x['importance'], reverse=True):
                                    rprint(f"{row['feature']}: {row['importance']:.4f}")

                            # Print task importance for each configuration
                            config_tasks = [
                                row for row in task_importance_all 
                                if row['window_size'] == ws and row['stride'] == st
                            ]
                            if config_tasks:
                                rprint(f"\n[bold blue]Task Importance for window_size={ws}, stride={st}:[/bold blue]")
                                for row in sorted(config_tasks, key=lambda x: x['importance'], reverse=True):
                                    rprint(f"Task {row['task_id']}: {row['importance']:.4f}")

            except Exception as e:
                rprint(f"[red]Error saving results to S3: {str(e)}[/red]")

        cleanup_wandb()
        rprint("[bold green]Handwriting Analysis completed successfully![/bold green]")
        
    except Exception as e:
        cleanup_wandb()
        rprint(f"[red]Error in main function: {str(e)}[/red]")
        raise e


if __name__ == "__main__":
    main()
