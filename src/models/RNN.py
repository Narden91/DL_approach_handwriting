import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, MatthewsCorrCoef, Precision, Recall, F1Score
from rich import print as rprint

class RNNDebugger:
    @staticmethod
    def check_tensor(tensor: torch.Tensor, name: str, step: str):
        if tensor is None:
            rprint(f"[red]{step} - {name} is None[/red]")
            return False
        
        is_nan = torch.isnan(tensor).any()
        is_inf = torch.isinf(tensor).any()
        
        if is_nan or is_inf:
            rprint(f"[red]{step} - {name} contains NaN: {is_nan}, Inf: {is_inf}[/red]")
            return False
            
        stats = {
            'min': float(tensor.min()),
            'max': float(tensor.max()),
            'mean': float(tensor.mean()),
            'std': float(tensor.std())
        }
        
        rprint(f"[green]{step} - {name} stats: {stats}[/green]")
        return True

    @staticmethod
    def check_gradients(model: nn.Module, step: str):
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                param_norm = param.norm()
                
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    rprint(f"[red]{step} - Gradient NaN/Inf detected in {name}[/red]")
                    return False
                    
                rprint(f"[blue]{step} - {name} - grad norm: {grad_norm:.4f}, param norm: {param_norm:.4f}[/blue]")
        return True

class RNN(pl.LightningModule):
    def __init__(self, input_size: int = 13, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.save_hyperparameters()
        
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.val_precision = Precision(task='binary')
        self.val_recall = Recall(task='binary')
        self.val_f1 = F1Score(task='binary')
        self.val_mcc = MatthewsCorrCoef(task='binary')
        
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-5)
        self.input_norm = nn.BatchNorm1d(input_size, eps=1e-5, momentum=0.1)
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        self.classifier = nn.Linear(hidden_size * 2, 1)
        
        rprint(f"[green]Initialized RNN with input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}[/green]")
    
    def forward(self, x, task_ids, masks): 
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        x = torch.clamp(x, -10, 10)
        x = self.input_norm(x.transpose(1,2)).transpose(1,2)
        outputs, _ = self.rnn(x)
        outputs = F.relu(outputs)
        outputs = F.dropout(outputs, p=0.1, training=self.training)
        return self.classifier(outputs[:, -1, :])
    
    def debug_forward(self, features, task_ids, masks):
        try:
            # Check input tensors
            self.debugger.check_tensor(features, "Input Features", "Pre-Forward")
            self.debugger.check_tensor(masks, "Masks", "Pre-Forward")
            
            # Check batch normalization
            x = self.input_norm(features.transpose(1,2)).transpose(1,2)
            self.debugger.check_tensor(x, "After BatchNorm", "Forward")
            
            # Check RNN outputs
            outputs, _ = self.rnn(x)
            self.debugger.check_tensor(outputs, "RNN Outputs", "Forward")
            
            # Check final output
            logits = self.classifier(outputs[:, -1, :])
            self.debugger.check_tensor(logits, "Logits", "Forward")
            
            return True
            
        except Exception as e:
            rprint(f"[red]Error in forward pass: {str(e)}[/red]")
            return False
    
    def training_step(self, batch, batch_idx):
        features, labels, task_ids, masks = batch
        features = torch.where(torch.isnan(features), torch.zeros_like(features), features)
        logits = self(features, task_ids, masks)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        if not torch.isnan(loss):
            self.log('train_loss', loss)
            self.train_acc(torch.sigmoid(logits), labels)
            self.log('train_acc', self.train_acc)
            return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.model_config['learning_rate'],
            weight_decay=self.model_config['weight_decay']
        )
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        features, labels, task_ids, masks = batch
        logits = self(features, task_ids, masks)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        preds = torch.sigmoid(logits)
        self.val_acc(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)
        self.val_mcc(preds, labels)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_precision', self.val_precision)
        self.log('val_recall', self.val_recall)
        self.log('val_f1', self.val_f1)
        self.log('val_mcc', self.val_mcc)
        
        return loss