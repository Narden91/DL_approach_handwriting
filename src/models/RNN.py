import numpy as np
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
    def __init__(self, input_size=13, hidden_size=128, num_layers=2, 
                 num_tasks=34, task_embedding_dim=32, nonlinearity="tanh", 
                 dropout=0.1, verbose=False):
        super().__init__()
        
        self.input_norm = nn.BatchNorm1d(input_size, eps=1e-5, momentum=0.1)
        self.task_embedding = nn.Embedding(num_tasks+1, task_embedding_dim)  # +1 for padding
        
        # Combine feature dimensions with task embeddings
        rnn_input_size = input_size + task_embedding_dim
        
        self.rnn = nn.RNN(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.classifier = nn.Linear(hidden_size * 2, 1)
        self._init_weights()
        self.verbose = verbose

    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                
    def set_class_weights(self, dataset):
        if hasattr(dataset, 'class_weights'):
            self.class_weights = torch.tensor([
                dataset.class_weights[0],
                dataset.class_weights[1]
            ], device=self.device)

    def forward(self, x, task_ids, masks):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, num_features)
            task_ids: Tensor of shape (batch_size, 1)
            masks: Tensor of shape (batch_size, seq_len)
        """
        
        # Less aggressive normalization
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        x = torch.clamp(x, -100, 100)  # Wider range
        
        # Add residual connection
        identity = x
        x = self.input_norm(x.transpose(1,2)).transpose(1,2)
        x = x + identity  # Residual connection
        
        # Process task embeddings
        task_ids = task_ids.squeeze(-1)
        task_emb = self.task_embedding(task_ids)
        task_emb = task_emb.unsqueeze(1).repeat(1, x.size(1), 1)
        
        # Combine features with task embeddings
        x = torch.cat([x, task_emb], dim=-1)
        
        # Apply masking
        if masks is not None:
            x = x * masks.unsqueeze(-1)
        
        # RNN forward pass
        outputs, _ = self.rnn(x)
        outputs = F.relu(outputs)
        outputs = F.dropout(outputs, p=0.1, training=self.training)
        
        return self.classifier(outputs[:, -1, :])

    def aggregate_predictions(self, window_preds, subject_ids):
        """Aggregate window-level predictions to subject-level"""
        pred_probs = torch.sigmoid(window_preds).cpu().numpy()
        subject_preds = {}
        
        for pred, subj_id in zip(pred_probs, subject_ids):
            if subj_id not in subject_preds:
                subject_preds[subj_id] = []
            subject_preds[subj_id].append(pred)
        
        # Average predictions for each subject
        final_preds = {subj: np.mean(preds) > 0.5 
                      for subj, preds in subject_preds.items()}
        return final_preds