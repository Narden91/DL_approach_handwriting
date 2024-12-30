import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchmetrics import Accuracy, MatthewsCorrCoef, Precision, Recall, F1Score
from rich import print as rprint

from src.models.base import BaseModel

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

class RNN(BaseModel):
    def __init__(self, input_size=13, hidden_size=128, num_layers=2, nonlinearity="tanh", dropout=0.1, verbose=False):
        super().__init__()
        # self.save_hyperparameters()
        
        self.input_norm = nn.BatchNorm1d(input_size, eps=1e-5, momentum=0.1)
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity='relu',
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.classifier = nn.Linear(hidden_size * 2, 1)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x, task_ids, masks):
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        x = torch.clamp(x, -10, 10)
        x = self.input_norm(x.transpose(1,2)).transpose(1,2)
        
        outputs, _ = self.rnn(x)
        outputs = F.relu(outputs)
        outputs = F.dropout(outputs, p=0.1, training=self.training)
        return self.classifier(outputs[:, -1, :])
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.model_config['learning_rate'],
            weight_decay=self.model_config['weight_decay']
        )
        return optimizer