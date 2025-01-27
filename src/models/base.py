import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from torchmetrics import Accuracy, MatthewsCorrCoef, Precision, Recall, F1Score, Specificity
import numpy as np


class MetricHandler:
    """Handles metric computation and storage for model evaluation.
    
    This class ensures all metrics are on the correct device and handles
    device management for metric calculations.
    """
    def __init__(self, prefix: str, device: Optional[torch.device] = None):
        # Create metrics
        metrics = {
            'acc': Accuracy(task='binary', num_classes=2),
            'precision': Precision(task='binary', num_classes=2),
            'recall': Recall(task='binary', num_classes=2),
            'specificity': Specificity(task='binary'),
            'f1': F1Score(task='binary'),
            'mcc': MatthewsCorrCoef(task='binary')
        }
        
        # Move metrics to the specified device
        self.metrics = {}
        for name, metric in metrics.items():
            if device is not None:
                metric = metric.to(device)
            self.metrics[name] = metric
            
        self.prefix = prefix
        self.device = device
        
    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Update metrics ensuring tensors are on the correct device.
        
        Args:
            preds: Model predictions
            labels: Ground truth labels
            
        Returns:
            Dictionary of computed metrics
        """
        # Ensure tensors are on the correct device
        if self.device is not None:
            preds = preds.to(self.device)
            labels = labels.to(self.device)
        
        # Compute metrics
        computed_metrics = {}
        for name, metric in self.metrics.items():
            metric_value = metric(preds, labels)
            computed_metrics[f"{self.prefix}_{name}"] = metric_value
            
        return computed_metrics


class BaseModel(pl.LightningModule):
    """Base model class implementing common functionality for all models.
    
    This class provides the foundation for all model implementations, handling
    common operations like metric tracking, optimization, and loss calculation.
    """
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.optimal_threshold = 0.5
        
        # Initialize metric handlers
        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None
        
        # Track predictions and labels for analysis
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize metrics on the correct device during setup."""
        if self.train_metrics is None:
            # Get the device from the model parameters
            device = next(self.parameters()).device
            
            # Initialize metrics with the correct device
            self.train_metrics = MetricHandler('train', device)
            self.val_metrics = MetricHandler('val', device)
            self.test_metrics = MetricHandler('test', device)
    
    def weighted_binary_cross_entropy(self, logits: torch.Tensor, 
                                    labels: torch.Tensor) -> torch.Tensor:
        """Enhanced weighted BCE loss with focal loss component.
        
        Args:
            logits: Raw model outputs
            labels: Ground truth labels
            
        Returns:
            Computed loss value
        """
        labels = labels.float()
        
        # Calculate focal weights
        probs = torch.sigmoid(logits)
        pt = torch.where(labels == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** 2  # gamma = 2
        
        # Apply class weights if available
        if hasattr(self, 'class_weights') and self.class_weights is not None:
            pos_weight = self.class_weights[1]
            neg_weight = self.class_weights[0]
            class_weights = torch.where(labels == 1, 
                                      torch.tensor(pos_weight).to(labels.device),
                                      torch.tensor(neg_weight).to(labels.device))
        else:
            class_weights = torch.ones_like(labels)
        
        # Combine weights and calculate loss
        weights = focal_weight * class_weights
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, labels, reduction='none'
        )
        
        return (bce_loss * weights).mean()
    
    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """Execute training step with proper device management."""
        features, labels, task_ids, masks = batch
        # Move input tensors to the model's device
        features = features.to(self.device)
        labels = labels.to(self.device)
        task_ids = task_ids.to(self.device)
        masks = masks.to(self.device)
        
        # Handle NaN values
        features = torch.where(torch.isnan(features), torch.zeros_like(features), features)
        
        # Forward pass
        logits = self(features, task_ids, masks)
        loss = self.weighted_binary_cross_entropy(logits, labels)
        preds = torch.sigmoid(logits)
        
        # Update metrics (they are already on the correct device)
        metrics = self.train_metrics.update(preds, labels)
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)
        
        self.training_step_outputs.append({
            'loss': loss,
            'preds': preds.detach(),
            'labels': labels.detach()
        })
        
        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Execute validation step with proper device management."""
        features, labels, task_ids, masks = batch
        # Move input tensors to the model's device
        features = features.to(self.device)
        labels = labels.to(self.device)
        task_ids = task_ids.to(self.device)
        masks = masks.to(self.device)
        
        # Forward pass
        logits = self(features, task_ids, masks)
        loss = self.weighted_binary_cross_entropy(logits, labels)
        preds = torch.sigmoid(logits)
        
        # Update metrics (they are already on the correct device)
        metrics = self.val_metrics.update(preds, labels)
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)
        
        output = {
            'loss': loss,
            'preds': preds.detach(),
            'labels': labels.detach()
        }
        
        self.validation_step_outputs.append(output)
        return output
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.model_config['learning_rate'],
            weight_decay=self.model_config['weight_decay'],
            amsgrad=True
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1",
                "frequency": 1
            }
        }
    
    def aggregate_predictions(self, window_preds: torch.Tensor, 
                            subject_ids: np.ndarray) -> Dict[int, bool]:
        """Aggregate window-level predictions to subject-level with confidence weighting.
        
        Args:
            window_preds: Predictions for each window
            subject_ids: Corresponding subject IDs
            
        Returns:
            Dictionary mapping subject IDs to final predictions
        """
        pred_probs = torch.sigmoid(window_preds).cpu().numpy()
        subject_preds = {}
        
        for pred, subj_id in zip(pred_probs, subject_ids):
            if subj_id not in subject_preds:
                subject_preds[subj_id] = []
            subject_preds[subj_id].append(pred)
        
        # Weighted average based on prediction confidence
        final_preds = {}
        for subj, preds in subject_preds.items():
            preds = np.array(preds)
            confidence = np.abs(preds - 0.5) + 0.5
            weighted_avg = np.average(preds, weights=confidence)
            final_preds[subj] = weighted_avg > 0.5
            
        return final_preds