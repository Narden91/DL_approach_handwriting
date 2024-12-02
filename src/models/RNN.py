from typing import Dict, Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score
from rich import print as rprint


class RNN(pl.LightningModule):
    def __init__(self, input_size: int = 13, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.save_hyperparameters()
        
        # Metrics
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.val_precision = Precision(task='binary')
        self.val_recall = Recall(task='binary')
        self.val_f1 = F1Score(task='binary')
        
        # RNN layers
        self.input_norm = nn.BatchNorm1d(input_size)
        self.rnn = nn.RNN(input_size=input_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         batch_first=True,
                         dropout=0.1)
        
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
        
        self.classifier = nn.Linear(hidden_size, 1)
        rprint(f"[green]Initialized RNN with input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}[/green]")
    
    def forward(self, x, task_ids, masks):
        x = self.input_norm(x.transpose(1,2)).transpose(1,2)
        outputs, _ = self.rnn(x)
        return self.classifier(outputs[:, -1, :])
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.model_config['learning_rate'],
            weight_decay=self.model_config['weight_decay']
        )
        return optimizer
    
    def training_step(self, batch, batch_idx):
        features, labels, task_ids, masks = batch
        logits = self(features, task_ids, masks)
        
        # Debug prints
        print(f"Labels: {labels.float().mean():.3f}")
        print(f"Logits before sigmoid: {logits.mean():.3f}")
        
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        print(f"Loss: {loss.item():.3f}")
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
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
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_precision', self.val_precision)
        self.log('val_recall', self.val_recall)
        self.log('val_f1', self.val_f1)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)