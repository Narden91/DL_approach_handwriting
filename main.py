import sys

import pandas as pd

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
from src.models.RNN import RNN
from src.utils.print_info import check_cuda_availability, print_dataset_info, print_feature_info


def set_global_seed(seed: int) -> None:
    """Set seed for reproducibility across all libraries."""
    import random
    import numpy as np
    import torch
    import pytorch_lightning as pl
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)


@hydra.main(version_base="1.1", config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    try:
        set_global_seed(cfg.seed)
        rprint("[bold blue]Starting Handwriting Analysis with 5-Fold Cross Validation[/bold blue]")
        check_cuda_availability()
        
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
                seed=cfg.seed
            )
            
            data_module.setup()
            
            print_feature_info(data_module) if cfg.verbose else None
            print_dataset_info(data_module) if cfg.verbose else None
                        
            model = RNN(input_size=data_module.get_feature_dim())
            model.model_config = {
                'learning_rate': cfg.training.learning_rate,
                'weight_decay': cfg.training.weight_decay
            }
            
            logger = pl.loggers.CSVLogger(
                save_dir="logs",
                name=f"fold_{fold}"
            )
            
            trainer = pl.Trainer(
                max_epochs=cfg.training.max_epochs,
                accelerator='gpu' if cfg.device == 'cuda' else 'cpu',
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=cfg.training.early_stopping_patience),
                    ModelCheckpoint(
                        dirpath=f"checkpoints/fold_{fold}",
                        filename=f"model_fold_{fold}",
                        monitor='val_loss'
                    )
                ],
                gradient_clip_val=cfg.training.gradient_clip_val,
                logger=logger
            )
            
            trainer.fit(model, data_module)
            
            fold_metrics.append({
                'fold': fold + 1,
                'val_loss': trainer.callback_metrics['val_loss'].item(),
                'val_acc': trainer.callback_metrics['val_acc'].item(),
                'val_f1': trainer.callback_metrics['val_f1'].item(),
                'val_mcc': trainer.callback_metrics['val_mcc'].item()
            })

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

