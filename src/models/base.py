from typing import Dict, Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score


class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.val_precision = Precision(task='binary')
        self.val_recall = Recall(task='binary')
        self.val_f1 = F1Score(task='binary')
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.model_config['learning_rate'],
            weight_decay=self.model_config['weight_decay']
        )
        return optimizer

    def forward(self, x, task_ids, masks):
        raise NotImplementedError("Subclass must implement forward method")
    
    def training_step(self, batch, batch_idx):
        features, labels, task_ids, masks = batch
        logits = self(features, task_ids, masks)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        self.train_acc(torch.sigmoid(logits), labels)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        features, labels, task_ids, masks = batch
        logits = self(features, task_ids, masks)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
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