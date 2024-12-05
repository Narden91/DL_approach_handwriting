from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

class TrainingVisualizer(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.epochs = []
        self.current_fold = 0
        self.current_epoch = 0
        
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_loss = metrics.get('train_loss', 0).item()
        train_acc = metrics.get('train_acc', 0).item()
        
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        
        self._update_plot()
        # self._print_metrics("Training", train_loss, train_acc)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_loss = metrics.get('val_loss', 0).item()
        val_acc = metrics.get('val_acc', 0).item()
        val_f1 = metrics.get('val_f1', 0).item()
        val_mcc = metrics.get('val_mcc', 0).item()
        
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        self.epochs.append(self.current_epoch)
        self.current_epoch += 1
        
        self._update_plot()
        # self._print_metrics("Validation", val_loss, val_acc, val_f1, val_mcc)
        
    def _print_metrics(self, phase, loss, acc, f1=None, mcc=None):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Phase", phase)
        table.add_row("Epoch", str(self.current_epoch))
        table.add_row("Loss", f"{loss:.4f}")
        table.add_row("Accuracy", f"{acc:.4f}")
        
        if f1 is not None:
            table.add_row("F1 Score", f"{f1:.4f}")
        if mcc is not None:
            table.add_row("MCC", f"{mcc:.4f}")
            
        rprint(Panel(table, title=f"[bold blue]Fold {self.current_fold + 1} - {phase} Metrics[/bold blue]"))
            
    def _update_plot(self):
        plt.figure(figsize=(12, 8))
        
        # Loss subplot
        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, label='Training Loss', marker='o')
        plt.plot(self.val_losses, label='Validation Loss', marker='o')
        plt.title(f'Metrics Evolution - Fold {self.current_fold + 1}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy subplot
        plt.subplot(2, 1, 2)
        plt.plot(self.train_accs, label='Training Accuracy', marker='o')
        plt.plot(self.val_accs, label='Validation Accuracy', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_dir = Path('plots')
        save_dir.mkdir(exist_ok=True)
        plt.savefig(save_dir / f'metrics_fold_{self.current_fold + 1}.png')
        plt.close()
        
    def on_fit_end(self, trainer, pl_module):
        self.current_fold += 1
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.epochs = []