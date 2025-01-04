from datetime import datetime
import sys

sys.dont_write_bytecode = True
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
from src.data.datamodule import HandwritingDataModule
from src.utils.callbacks import GradientMonitorCallback, ThresholdTuner
from s3_operations.s3_handler import config
from s3_operations.s3_io import S3IOHandler
from src.utils.config_operations import ConfigOperations
from src.utils.print_info import check_cuda_availability, print_dataset_info, print_feature_info, print_subject_metrics, process_metrics
from src.utils.majority_vote import get_predictions, compute_subject_metrics


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
        cfg = ConfigOperations.merge_configurations(cfg)
                        
        set_global_seed(cfg.seed)
        rprint("[bold blue]Starting Handwriting Analysis with 5-Fold Cross Validation[/bold blue]")
        check_cuda_availability(cfg.verbose)
        
        file_key_load: str = f"{cfg.data.s3_folder_input}/{cfg.data.data_filename}"
        output_filename: str = f"{cfg.data.output_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_key_save: str = f"{cfg.data.s3_folder_output}/{output_filename}"
        s3_handler: S3IOHandler = S3IOHandler(config, verbose=cfg.verbose)
        
        # Metrics for each fold
        fold_metrics = []
        
        for window_size in cfg.data.window_sizes:
            for stride in cfg.data.strides:
                rprint(f"\n[bold cyan]====== Testing window_size={window_size}, stride={stride} ======[/bold cyan]")
                for fold in range(cfg.num_folds):
                    rprint(f"\n[bold cyan]====== Starting Fold {fold + 1}/5 ======[/bold cyan]")
                    try:
                        fold_seed = cfg.seed + fold
                        
                        data_module = HandwritingDataModule(
                            s3_handler=s3_handler,
                            file_key=file_key_load,
                            batch_size=cfg.data.batch_size,
                            window_size=window_size,
                            stride=stride,
                            num_workers=cfg.data.num_workers,
                            num_tasks=cfg.data.num_tasks,
                            val_size=cfg.data.val_size,
                            test_size=cfg.data.test_size,
                            column_names=dict(cfg.data.columns),
                            fold=fold,
                            n_folds=cfg.num_folds,
                            scaler_type=cfg.data.scaler,
                            seed=fold_seed,
                            verbose=cfg.verbose
                        )

                        data_module.setup()
                        
                        if cfg.verbose:
                            print_feature_info(data_module)
                            print_dataset_info(data_module)

                        model = ModelFactory.create_model(cfg, data_module, window_size=window_size)
                        model.model_config = {
                            'learning_rate': cfg.training.learning_rate,
                            'weight_decay': cfg.training.weight_decay
                        }

                        # Set class weights if available
                        if hasattr(data_module.train_dataset, 'class_weights'):
                            model.set_class_weights(data_module.train_dataset)

                        # Configure training and fit the model
                        trainer_config, wandb_logger = configure_training(cfg)
                        trainer = pl.Trainer(**trainer_config)
                        trainer.fit(model, data_module)
                        
                        # Get window-level predictions
                        train_subjects, train_labels, train_preds = get_predictions(trainer, model, data_module.train_dataloader())
                        test_subjects, test_labels, test_preds = get_predictions(trainer, model, data_module.test_dataloader())
                        
                        # Compute subject-level metrics
                        train_subject_metrics = compute_subject_metrics(train_subjects, train_labels, train_preds, cfg.verbose)
                        test_subject_metrics = compute_subject_metrics(test_subjects, test_labels, test_preds, cfg.verbose)
                        
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
                        
                        # Safely log metrics to wandb
                        safe_wandb_log(wandb_logger, metrics, trainer.current_epoch)
                        
                        fold_metrics.append(metrics)
                        print_subject_metrics(train_subject_metrics, test_subject_metrics, fold, cfg.verbose)
                    except Exception as e:
                        rprint(f"[red]Error in fold {fold + 1}: {str(e)}[/red]")
                        if cfg.test_mode:
                            return
                        else:
                            continue
                        
                    if cfg.test_mode and fold == 0:
                        break
        
        if fold_metrics:
            metrics_df = pd.DataFrame(fold_metrics)
            process_metrics(metrics_df, window_size, stride, cfg)
            try:
                s3_handler.save_data(metrics_df, file_key_save)
                rprint(f"[green]Successfully saved results to S3[/green]")
            except Exception as e:
                rprint(f"[red]Error saving results to S3: {str(e)}[/red]")
        
    except Exception as e:
        rprint(f"[red]Error in main function: {str(e)}[/red]")
        raise e


if __name__ == "__main__":
    main()
