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
from src.models.GRU import GRU 
from src.models.XLSTM import XLSTM
from src.models.LSTM import LSTM
from src.utils.trainer_visualizer import TrainingVisualizer
from s3_operations.s3_handler import config
from s3_operations.s3_io import S3IOHandler
from src.utils.config_operations import ConfigOperations
from src.utils.print_info import check_cuda_availability, print_dataset_info, print_feature_info, print_sets_info, print_predictions, print_fold_completion


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
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    visualizer = TrainingVisualizer()
    progress_bar = pl.callbacks.progress.TQDMProgressBar(
        refresh_rate=1,
        process_position=0,
        leave=True
    )
    
    # Add Wandb logger
    wandb_logger = WandbLogger(
        project="handwriting_analysis",
        name=f"{config.model.type}_ws{config.data.window_sizes}_str{config.data.strides}",
    )
    wandb_logger.experiment.config.update(dict(config), allow_val_change=True)
    
    trainer_config = {
        "accelerator": "gpu",
        "devices": 1,
        "max_epochs": config.training.max_epochs,
        "gradient_clip_val": config.training.gradient_clip_val,
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
        "enable_progress_bar": True,
        "enable_model_summary": True,
        "strategy": "auto",
        "sync_batchnorm": False,
        # Use Wandb logger
        "logger": wandb_logger,
        "enable_checkpointing": False,
        "enable_model_summary": True
        # "enable_progress_bar": True,
        # "enable_model_summary": True,
        # "strategy": "auto",
        # "sync_batchnorm": False,
        # "logger": False,
        # "enable_checkpointing": False,
        # "enable_model_summary": True
    }
    return trainer_config


def get_predictions(trainer, model, dataloader):
    """Get predictions for a given dataloader using the trained model.
    
    Args:
    - trainer: PyTorch Lightning Trainer object
    - model: PyTorch Lightning Module object
    - dataloader: PyTorch DataLoader object
    
    Returns:
    - List of subjects
    - List of true labels
    - List of predicted labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_subjects = []
    with torch.no_grad():
        for batch in dataloader:
            features, labels, task_ids, masks = batch
            logits = model(features, task_ids, masks)
            preds = torch.sigmoid(logits).round().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_subjects.extend(task_ids.cpu().numpy())
    return all_subjects, all_labels, all_preds
            

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
                    else:
                        model = RNN(
                            input_size=data_module.get_feature_dim(),
                            hidden_size=cfg.model.hidden_size,
                            num_layers=cfg.model.num_layers,
                            nonlinearity=cfg.model.rnn_specific.nonlinearity,
                            dropout=cfg.model.dropout,
                            verbose=cfg.verbose
                        )

                    model.model_config = {
                        'learning_rate': cfg.training.learning_rate,
                        'weight_decay': cfg.training.weight_decay
                    }

                    # Configure training and fit the model
                    trainer_config = configure_training(cfg)
                    trainer = pl.Trainer(**trainer_config)
                    trainer.fit(model, data_module)
                    
                    # Save the validation metrics
                    val_metrics = {
                        'val_loss': trainer.callback_metrics['val_loss'].item(),
                        'val_acc': trainer.callback_metrics['val_acc'].item(),
                        'val_f1': trainer.callback_metrics['val_f1'].item(),
                        'val_mcc': trainer.callback_metrics['val_mcc'].item()
                    }
                    
                    # Test the model
                    trainer.test(model, datamodule=data_module, verbose=cfg.verbose)
                    
                    # Store metrics for current fold
                    fold_metrics.append({
                        'window_size': window_size,
                        'stride': stride,
                        'fold': fold + 1,
                        **val_metrics,
                        'test_loss': trainer.callback_metrics['test_loss'].item(),
                        'test_acc': trainer.callback_metrics['test_acc'].item(),
                        'test_f1': trainer.callback_metrics['test_f1'].item(),
                        'test_mcc': trainer.callback_metrics['test_mcc'].item()
                    })

                    if cfg.verbose:
                        # Get predictions for train and test sets
                        train_subjects, train_labels, train_preds = get_predictions(trainer, model, data_module.train_dataloader())
                        test_subjects, test_labels, test_preds = get_predictions(trainer, model, data_module.test_dataloader())
                        print_predictions(train_subjects, train_labels, train_preds, fold, "train")
                        print_predictions(test_subjects, test_labels, test_preds, fold, "test")
                        print_fold_completion(fold, trainer)
                        
                    if cfg.test_mode and fold == 0:
                        break
        
        metrics_df = pd.DataFrame(fold_metrics)
        rprint("\n[bold blue]Cross Validation Results:[/bold blue]")
        rprint(metrics_df.to_string())

        # Save results to CSV file on S3 bucket
        s3_handler.save_data(metrics_df, file_key_save)

        mean_metrics = metrics_df.groupby(['window_size', 'stride']).mean()
        std_metrics = metrics_df.groupby(['window_size', 'stride']).std()
        for (window_size, stride), metrics in mean_metrics.iterrows():
            rprint(f"\n[bold blue]Results for window_size={window_size}, stride={stride}:[/bold blue]")
            for metric in ['val_loss', 'val_acc', 'val_f1', 'val_mcc']:
                rprint(f"{metric}: {metrics[metric]:.4f} ± {std_metrics.loc[(window_size, stride), metric]:.4f}")
            for metric in ['test_loss', 'test_acc', 'test_f1', 'test_mcc']:
                rprint(f"{metric}: {metrics[metric]:.4f} ± {std_metrics.loc[(window_size, stride), metric]:.4f}")

    except Exception as e:
        rprint(f"[red]Error in main function: {str(e)}[/red]")
        raise e


if __name__ == "__main__":
    main()
