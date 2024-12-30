from typing import Dict, Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, MatthewsCorrCoef, Precision, Recall, F1Score


class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.val_precision = Precision(task='binary')
        self.val_recall = Recall(task='binary')
        self.val_f1 = F1Score(task='binary')
        self.val_mcc = MatthewsCorrCoef(task='binary')
        self.test_acc = Accuracy(task='binary')
        self.test_precision = Precision(task='binary')
        self.test_recall = Recall(task='binary')
        self.test_f1 = F1Score(task='binary')
        self.test_mcc = MatthewsCorrCoef(task='binary')

    def training_step(self, batch, batch_idx):
        features, labels, task_ids, masks = batch
        features = torch.where(torch.isnan(features), torch.zeros_like(features), features)
        logits = self(features, task_ids, masks)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        self.train_acc(torch.sigmoid(logits), labels)
        # Log only essential metrics for the progress bar
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_acc, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels, task_ids, masks = batch
        logits = self(features, task_ids, masks)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        preds = torch.sigmoid(logits)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.val_mcc(preds, labels)
        
        # Control which metrics appear in the progress bar
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_f1', self.val_f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_mcc', self.val_mcc, prog_bar=False, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        features, labels, task_ids, masks = batch
        logits = self(features, task_ids, masks)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        preds = torch.sigmoid(logits)
        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        self.test_mcc(preds, labels)
        
        # Log test metrics without printing the table
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_f1', self.test_f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_mcc', self.test_mcc, prog_bar=True, on_step=False, on_epoch=True)
        return loss