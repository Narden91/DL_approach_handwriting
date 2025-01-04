from typing import Dict, Any
import pytorch_lightning as pl
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, MatthewsCorrCoef, Precision, Recall, F1Score, Specificity, AveragePrecision
import numpy as np
from sklearn.metrics import precision_recall_curve, auc


# class BaseModel(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.optimal_threshold = 0.5  # Will be optimized during validation
#         self.regularization_strength = 0.01  # L2 regularization strength
#         self.train_acc = Accuracy(task='binary')
#         self.train_precision = Precision(task='binary', zero_division=0)
#         self.train_recall = Recall(task='binary', zero_division=0)  # Same as Sensitivity
#         self.train_specificity = Specificity(task='binary', zero_division=0)
#         self.train_f1 = F1Score(task='binary', zero_division=0)
#         self.train_mcc = MatthewsCorrCoef(task='binary', ignore_index=-1)
        
#         self.val_acc = Accuracy(task='binary')
#         self.val_precision = Precision(task='binary', zero_division=0)
#         self.val_recall = Recall(task='binary', zero_division=0)  # Same as Sensitivity
#         self.val_specificity = Specificity(task='binary', zero_division=0)
#         self.val_f1 = F1Score(task='binary', zero_division=0)
#         self.val_mcc = MatthewsCorrCoef(task='binary', ignore_index=-1)
        
#         self.test_acc = Accuracy(task='binary')
#         self.test_precision = Precision(task='binary', zero_division=0)
#         self.test_recall = Recall(task='binary', zero_division=0)  # Same as Sensitivity
#         self.test_specificity = Specificity(task='binary', zero_division=0)
#         self.test_f1 = F1Score(task='binary', zero_division=0)
#         self.test_mcc = MatthewsCorrCoef(task='binary', ignore_index=-1)
        
#         self.train_ap = AveragePrecision(task='binary')
#         self.val_ap = AveragePrecision(task='binary')
#         self.test_ap = AveragePrecision(task='binary')
        
#         # Initialize class weights as None
#         self.class_weights = None

#     def set_class_weights(self, class_counts: Dict[int, int]):
#         """
#         Set class weights based on the class distribution.
#         Args:
#             class_counts: Dictionary with class labels as keys and counts as values
#         """
#         total_samples = sum(class_counts.values())
#         n_classes = len(class_counts)
        
#         # Compute balanced weights
#         weights = {
#             cls: total_samples / (n_classes * count) 
#             for cls, count in class_counts.items()
#         }
        
#         # Convert to tensor
#         self.class_weights = torch.tensor([weights[0], weights[1]], 
#                                         dtype=torch.float32,
#                                         device=self.device)

#     def weighted_binary_cross_entropy(self, logits, labels):
#         """
#         Compute weighted binary cross entropy loss with L2 regularization.
#         Args:
#             logits: Model predictions (before sigmoid)
#             labels: True labels
#         """
#         # Compute weighted BCE loss
#         if self.class_weights is None:
#             bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
#         else:
#             weights = torch.where(labels == 1, 
#                                 self.class_weights[1], 
#                                 self.class_weights[0])
#             bce_loss = F.binary_cross_entropy_with_logits(
#                 logits, labels.float(), 
#                 reduction='none'
#             )
#             bce_loss = (bce_loss * weights).mean()
        
#         # Add L2 regularization
#         l2_reg = 0
#         for param in self.parameters():
#             l2_reg += torch.norm(param, 2)
        
#         return bce_loss + self.regularization_strength * l2_reg
    
#     def configure_optimizers(self):
#         """
#         Configure optimizer and learning rate scheduler.
#         This is defined in the base class as all models use the same optimization strategy.
#         Individual models can override if needed.
#         """
#         if not hasattr(self, 'model_config'):
#             raise AttributeError("model_config not found. Make sure to set model_config with learning_rate and weight_decay.")
            
#         optimizer = torch.optim.AdamW(
#             self.parameters(),
#             lr=self.model_config['learning_rate'],
#             weight_decay=self.model_config['weight_decay']
#         )
        
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer,
#             mode='max',  # We want to maximize AP
#             factor=0.5,
#             patience=5,
#             verbose=self.verbose if hasattr(self, 'verbose') else False
#         )
        
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "monitor": "val_ap"  # Monitoring Average Precision
#             }
#         }

#     def training_step(self, batch, batch_idx):
#         features, labels, task_ids, masks = batch
#         features = torch.where(torch.isnan(features), torch.zeros_like(features), features)
#         logits = self(features, task_ids, masks)
        
#         # Use weighted loss
#         loss = self.weighted_binary_cross_entropy(logits, labels)
        
#         # Calculate and log all training metrics
#         preds = torch.sigmoid(logits)
#         self.train_acc(preds, labels)
#         self.train_precision(preds, labels)
#         self.train_recall(preds, labels)
#         self.train_specificity(preds, labels)
#         self.train_f1(preds, labels)
#         self.train_mcc(preds, labels)
        
#         # Log metrics - only accuracy and F1 on progress bar
#         self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
#         self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
#         self.log('train_precision', self.train_precision, prog_bar=False, on_step=False, on_epoch=True)
#         self.log('train_recall', self.train_recall, prog_bar=False, on_step=False, on_epoch=True)
#         self.log('train_specificity', self.train_specificity, prog_bar=False, on_step=False, on_epoch=True)
#         self.log('train_f1', self.train_f1, prog_bar=False, on_step=False, on_epoch=True)
#         self.log('train_mcc', self.train_mcc, prog_bar=False, on_step=False, on_epoch=True)
        
#         return loss

#     def find_optimal_threshold(self, logits, labels):
#         """Find optimal threshold that maximizes validation F1-score."""
#         probas = torch.sigmoid(logits).cpu().numpy()
#         labels = labels.cpu().numpy()
        
#         thresholds = np.linspace(0.3, 0.7, 20)
#         f1_scores = []
        
#         for threshold in thresholds:
#             preds = (probas > threshold).astype(int)
#             # Handle edge cases with zero_division parameter
#             f1 = f1_score(labels, preds, zero_division=0)
#             f1_scores.append(f1)
        
#         # If all F1 scores are zero, return default threshold
#         if np.all(np.array(f1_scores) == 0):
#             return 0.5
            
#         optimal_threshold = thresholds[np.argmax(f1_scores)]
#         return optimal_threshold

#     def validation_step(self, batch, batch_idx):
#         features, labels, task_ids, masks = batch
#         logits = self(features, task_ids, masks)
#         loss = self.weighted_binary_cross_entropy(logits, labels)
        
#         # Find optimal threshold and update predictions
#         self.optimal_threshold = self.find_optimal_threshold(logits.detach(), labels)
#         preds = torch.sigmoid(logits) > self.optimal_threshold
#         probs = torch.sigmoid(logits)
        
#         # Update all metrics including new AP
#         self.val_ap(probs, labels)
#         self.val_acc(probs, labels)
#         self.val_precision(probs, labels)
#         self.val_recall(probs, labels)
#         self.val_specificity(probs, labels)
#         self.val_f1(probs, labels)
#         self.val_mcc(probs, labels)
        
#         # Log metrics including AP
#         self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
#         self.log('val_ap', self.val_ap, prog_bar=True, on_step=False, on_epoch=True)
#         self.log('val_acc', self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
#         self.log('val_precision', self.val_precision, prog_bar=False, on_step=False, on_epoch=True)
#         self.log('val_recall', self.val_recall, prog_bar=False, on_step=False, on_epoch=True)
#         self.log('val_specificity', self.val_specificity, prog_bar=False, on_step=False, on_epoch=True)
#         self.log('val_f1', self.val_f1, prog_bar=False, on_step=False, on_epoch=True)
#         self.log('val_mcc', self.val_mcc, prog_bar=False, on_step=False, on_epoch=True)
        
#         return loss
    
#     def test_step(self, batch, batch_idx):
#         features, labels, task_ids, masks = batch
#         logits = self(features, task_ids, masks)
#         loss = self.weighted_binary_cross_entropy(logits, labels)
        
#         # Calculate and log all test metrics
#         preds = torch.sigmoid(logits)
#         self.test_acc(preds, labels)
#         self.test_precision(preds, labels)
#         self.test_recall(preds, labels)
#         self.test_specificity(preds, labels)
#         self.test_f1(preds, labels)
#         self.test_mcc(preds, labels)
        
#         # Log metrics - only accuracy and F1 on progress bar
#         self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
#         self.log('test_acc', self.test_acc, prog_bar=True, on_step=False, on_epoch=True)
#         self.log('test_precision', self.test_precision, prog_bar=False, on_step=False, on_epoch=True)
#         self.log('test_recall', self.test_recall, prog_bar=False, on_step=False, on_epoch=True)
#         self.log('test_specificity', self.test_specificity, prog_bar=False, on_step=False, on_epoch=True)
#         self.log('test_f1', self.test_f1, prog_bar=False, on_step=False, on_epoch=True)
#         self.log('test_mcc', self.test_mcc, prog_bar=False, on_step=False, on_epoch=True)
        
#         return loss

class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.optimal_threshold = 0.5
        
        # Initialize metrics as None - they will be set up during setup phase
        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None
        
        # Track predictions and labels for analysis
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def _init_metrics(self, prefix, device=None):
        """Initialize metrics and move them to the correct device"""
        metrics = {
            'acc': Accuracy(task='binary', num_classes=2),
            'precision': Precision(task='binary', num_classes=2),
            'recall': Recall(task='binary', num_classes=2),
            'specificity': Specificity(task='binary'),
            'f1': F1Score(task='binary'),
            'mcc': MatthewsCorrCoef(task='binary')
        }
        
        if device is not None:
            metrics = {k: metric.to(device) for k, metric in metrics.items()}
            
        return metrics
    
    def setup(self, stage=None):
        """Initialize metrics on the correct device during setup"""
        if self.train_metrics is None:
            device = next(self.parameters()).device
            self.train_metrics = self._init_metrics('train', device)
            self.val_metrics = self._init_metrics('val', device)
            self.test_metrics = self._init_metrics('test', device)
    
    def _log_metrics(self, preds, labels, stage='train'):
        """Log metrics ensuring everything is on the correct device"""
        metrics = getattr(self, f'{stage}_metrics')
        device = next(self.parameters()).device
        
        # Ensure inputs are on the correct device
        preds = preds.to(device)
        labels = labels.to(device)
        
        # Update and log all metrics
        computed_metrics = {}
        for name, metric in metrics.items():
            metric.to(device)  # Ensure metric is on correct device
            value = metric(preds, labels)
            self.log(f'{stage}_{name}', value, prog_bar=True)
            computed_metrics[name] = value.item()
            
        return computed_metrics

    def training_step(self, batch, batch_idx):
        features, labels, task_ids, masks = batch
        features = torch.where(torch.isnan(features), torch.zeros_like(features), features)
        logits = self(features, task_ids, masks)
        
        # Calculate weighted loss
        loss = self.weighted_binary_cross_entropy(logits, labels)
        
        # Get predictions
        preds = torch.sigmoid(logits)
        
        # Log metrics
        metrics = self._log_metrics(preds, labels, 'train')
        
        # Store step output
        self.training_step_outputs.append({
            'loss': loss,
            'preds': preds.detach(),
            'labels': labels.detach(),
            'metrics': metrics
        })
        
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels, task_ids, masks = batch
        logits = self(features, task_ids, masks)
        
        # Calculate loss
        loss = self.weighted_binary_cross_entropy(logits, labels)
        
        # Get predictions
        preds = torch.sigmoid(logits)
        
        # Log metrics
        metrics = self._log_metrics(preds, labels, 'val')
        
        # Store and return step output
        output = {
            'loss': loss,
            'preds': preds.detach(),  # Return predictions directly
            'labels': labels.detach(),
            'metrics': metrics
        }
        
        self.validation_step_outputs.append(output)
        return output

    def on_train_epoch_end(self):
        if not self.training_step_outputs:
            return
            
        # Ensure everything is on the same device
        device = next(self.parameters()).device
        all_preds = torch.cat([x['preds'].to(device) for x in self.training_step_outputs])
        all_labels = torch.cat([x['labels'].to(device) for x in self.training_step_outputs])
        
        # Find optimal threshold
        thresholds = torch.linspace(0.1, 0.9, 81, device=device)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            binary_preds = (all_preds > threshold).float()
            f1 = F1Score(task='binary').to(device)(binary_preds, all_labels)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        self.log('optimal_threshold', self.optimal_threshold)
        
        # Clear saved outputs
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
            
        # Ensure everything is on the same device
        device = next(self.parameters()).device
        all_preds = torch.cat([x['preds'].to(device) for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'].to(device) for x in self.validation_step_outputs])
        
        # Log final validation metrics with optimal threshold
        binary_preds = (all_preds > self.optimal_threshold).float()
        _ = self._log_metrics(binary_preds, all_labels, 'val')
        
        # Clear saved outputs
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.model_config['learning_rate'],
            weight_decay=self.model_config['weight_decay'],
            amsgrad=True  # Add this for better optimization
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

    def weighted_binary_cross_entropy(self, logits, labels):
        """Enhanced weighted BCE loss with focal loss component"""
        # Convert labels to float
        labels = labels.float()
        
        # Calculate focal weights
        probs = torch.sigmoid(logits)
        pt = torch.where(labels == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** 2  # gamma = 2
        
        # Get class weights
        if hasattr(self, 'class_weights') and self.class_weights is not None:
            pos_weight = self.class_weights[1]
            neg_weight = self.class_weights[0]
            class_weights = torch.where(labels == 1, 
                                      torch.tensor(pos_weight).to(labels.device),
                                      torch.tensor(neg_weight).to(labels.device))
        else:
            class_weights = torch.ones_like(labels)
        
        # Combine focal and class weights
        weights = focal_weight * class_weights
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, labels, reduction='none'
        )
        
        # Apply weights
        weighted_loss = (bce_loss * weights).mean()
        
        return weighted_loss