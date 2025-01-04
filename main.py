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
from src.data.datamodule import HandwritingDataModule
from src.models.RNN import RNN
from src.models.simpleRNN import SimpleRNN
from src.models.GRU import GRU 
from src.models.XLSTM import XLSTM
from src.models.LSTM import LSTM
from src.utils.trainer_visualizer import TrainingVisualizer
from src.utils.callbacks import GradientMonitorCallback, ThresholdTuner
from s3_operations.s3_handler import config
from s3_operations.s3_io import S3IOHandler
from src.utils.config_operations import ConfigOperations
from src.utils.print_info import check_cuda_availability, print_dataset_info, print_feature_info, print_subject_metrics
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
    
    # Initialize Wandb logger
    wandb_logger = WandbLogger(
        project="handwriting_analysis",
        name=f"{config.model.type}_ws{config.data.window_sizes}_str{config.data.strides}",
    )
    
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
        "logger": wandb_logger,
        "log_every_n_steps": 10,
        "val_check_interval": 0.5,
        "enable_checkpointing": False,
        "deterministic": True
    }
    
    return trainer_config, wandb_logger
            

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

                    if cfg.model.type.lower() == "lstm":
                        model = LSTM(
                            input_size=data_module.get_feature_dim(),
                            hidden_size=cfg.model.hidden_size,
                            num_layers=cfg.model.num_layers,
                            dropout=cfg.model.dropout,
                            layer_norm=cfg.model.lstm_specific.layer_norm,
                            verbose=cfg.verbose
                        )
                    elif cfg.model.type.lower() == "xlstm":
                        model = XLSTM(
                            input_size=data_module.get_feature_dim(),
                            hidden_size=cfg.model.hidden_size,
                            num_layers=cfg.model.num_layers,
                            dropout=cfg.model.dropout,
                            layer_norm=cfg.model.lstm_specific.layer_norm,
                            recurrent_dropout=cfg.model.xlstm_specific.recurrent_dropout,
                            verbose=cfg.verbose
                        )
                    elif cfg.model.type.lower() == "gru":
                        model = GRU(
                            input_size=data_module.get_feature_dim(),
                            hidden_size=cfg.model.hidden_size,
                            num_layers=cfg.model.num_layers,
                            dropout=cfg.model.dropout,
                            batch_first=cfg.model.gru_specific.batch_first,
                            bidirectional=cfg.model.bidirectional,
                            bias=cfg.model.gru_specific.bias,
                            verbose=cfg.verbose
                        )
                    elif cfg.model.type.lower() == "simplernn":
                        model = SimpleRNN(
                            input_size=data_module.get_feature_dim(),
                            hidden_size=cfg.model.hidden_size,
                            task_embedding_dim=cfg.model.task_embedding_dim,
                            num_tasks=cfg.data.num_tasks,
                            dropout=cfg.model.dropout,
                            embedding_dropout=cfg.model.embedding_dropout,
                            zoneout_prob=cfg.model.zoneout_prob,
                            activity_l1=cfg.model.activity_l1,
                            verbose=cfg.verbose
                        )
                    else:
                        model = RNN(
                            input_size=data_module.get_feature_dim(),
                            hidden_size=cfg.model.hidden_size,
                            num_layers=cfg.model.num_layers,
                            num_tasks=cfg.data.num_tasks,
                            task_embedding_dim=cfg.model.task_embedding_dim,
                            nonlinearity=cfg.model.rnn_specific.nonlinearity,
                            dropout=cfg.model.dropout,
                            verbose=cfg.verbose
                        )

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
                    
                    # Log metrics to wandb
                    wandb_logger.log_metrics({
                        **metrics,
                        'epoch': trainer.current_epoch
                    })
                    
                    fold_metrics.append(metrics)
                    print_subject_metrics(train_subject_metrics, test_subject_metrics, fold, cfg.verbose)
                    
                    if cfg.test_mode and fold == 0:
                        break
        
        # Save and display results
        metrics_df = pd.DataFrame(fold_metrics)
        
        # Save to S3
        s3_handler.save_data(metrics_df, file_key_save)
        
        # Display aggregated results with comprehensive metrics
        mean_metrics = metrics_df.groupby(['window_size', 'stride']).mean()
        std_metrics = metrics_df.groupby(['window_size', 'stride']).std()
        
        for (window_size, stride), metrics in mean_metrics.iterrows():
            rprint(f"\n[bold blue]Results for window_size={window_size}, stride={stride}:[/bold blue]")
            
            metric_groups = {
                'Training Metrics': 'train_subject_',
                'Testing Metrics': 'test_subject_'
            }
            
            for group_name, prefix in metric_groups.items():
                rprint(f"\n[bold cyan]{group_name}:[/bold cyan]")
                for metric in ['acc', 'precision', 'recall', 'specificity', 'f1', 'mcc']:
                    metric_name = f"{prefix}{metric}"
                    if metric_name in metrics:
                        mean_val = metrics[metric_name]
                        std_val = std_metrics.loc[(window_size, stride), metric_name]
                        rprint(f"{metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")

    except Exception as e:
        rprint(f"[red]Error in main function: {str(e)}[/red]")
        raise e


if __name__ == "__main__":
    main()
