from typing import Dict, Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score


class BaseModel(pl.LightningModule):
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        
        # Initialize metrics
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.val_precision = Precision(task='binary')
        self.val_recall = Recall(task='binary')
        self.val_f1 = F1Score(task='binary')
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.trainer.model.training.learning_rate,
            weight_decay=self.trainer.model.training.weight_decay
        )
        return optimizer
    
    def training_step(self, batch, batch_idx):
        features, labels, task_ids = batch
        logits = self(features, task_ids)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        # Log metrics
        self.train_acc(torch.sigmoid(logits), labels)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        features, labels, task_ids = batch
        logits = self(features, task_ids)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        # Log metrics
        preds = torch.sigmoid(logits)
        self.val_acc(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)
        
        self.log('val_loss', loss)
        self.log('val_acc', self.val_acc)
        self.log('val_precision', self.val_precision)
        self.log('val_recall', self.val_recall)
        self.log('val_f1', self.val_f1)
        
        return loss