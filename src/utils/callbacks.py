import pytorch_lightning as pl
from torchmetrics import Accuracy, MatthewsCorrCoef, Precision, Recall, F1Score, Specificity
import torch


class GradientMonitorCallback(pl.Callback):
    def on_after_backward(self, trainer, model):
        # Monitor gradient norms
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        if total_norm > 10:
            trainer.logger.experiment.log({"gradient_norm": total_norm})
            


class ThresholdTuner(pl.Callback):
    def __init__(self):
        super().__init__()
        self.val_preds = []
        self.val_labels = []
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Collect predictions and labels from validation batches"""
        _, labels, _, _ = batch
        
        # Handle outputs whether it's a tensor or dictionary
        if isinstance(outputs, dict) and 'loss' in outputs:
            preds = outputs['preds']  # Get predictions directly
        else:
            preds = outputs  # Assume outputs is the predictions tensor
            
        self.val_preds.append(preds.detach())
        self.val_labels.append(labels.detach())
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Find optimal threshold using validation data"""
        if not self.val_preds:
            return
        
        # Ensure everything is on the same device
        device = pl_module.device
        
        # Concatenate all predictions and labels
        all_preds = torch.cat([x.to(device) for x in self.val_preds])
        all_labels = torch.cat([x.to(device) for x in self.val_labels])
        
        # Try different thresholds
        best_threshold = 0.5
        best_f1 = 0
        best_metrics = None
        
        for threshold in torch.linspace(0.1, 0.9, 81, device=device):
            binary_preds = (all_preds > threshold).float()
            
            # Calculate metrics
            metrics = {
                'accuracy': Accuracy(task='binary', num_classes=2).to(device)(binary_preds, all_labels).item(),
                'precision': Precision(task='binary', num_classes=2).to(device)(binary_preds, all_labels).item(),
                'recall': Recall(task='binary', num_classes=2).to(device)(binary_preds, all_labels).item(),
                'specificity': Specificity(task='binary').to(device)(binary_preds, all_labels).item(),
                'f1': F1Score(task='binary').to(device)(binary_preds, all_labels).item(),
                'mcc': MatthewsCorrCoef(task='binary').to(device)(binary_preds, all_labels).item()
            }
            
            # Use F1 score for threshold selection
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_threshold = threshold
                best_metrics = metrics
        
        # Update model's threshold
        pl_module.optimal_threshold = best_threshold
        
        # Log metrics with best threshold
        pl_module.log('best_threshold', best_threshold)
        for name, value in best_metrics.items():
            pl_module.log(f'best_val_{name}', value)
        
        # Clear collected predictions
        self.val_preds.clear()
        self.val_labels.clear()