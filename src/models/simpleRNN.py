import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print as rprint

from src.models.base import BaseModel


class VariationalDropout(nn.Module):
    """Applies the same dropout mask across time steps."""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        
        mask = torch.bernoulli(
            (1.0 - self.p) * torch.ones_like(x[:, 0, :])
        ).unsqueeze(1) / (1.0 - self.p)
        
        return mask * x

class LayerNormRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, nonlinearity="tanh"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        
        # Regular linear layers
        self.ih = nn.Linear(input_size, hidden_size)
        self.hh = nn.Linear(hidden_size, hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using orthogonal initialization."""
        nn.init.orthogonal_(self.ih.weight, gain=0.1)
        nn.init.orthogonal_(self.hh.weight, gain=0.1)
        nn.init.zeros_(self.ih.bias)
        nn.init.zeros_(self.hh.bias)
        
    def forward(self, input, hidden):
        # Combined input and hidden transformation
        gates = self.ih(input) + self.hh(hidden)
        gates = self.layer_norm(gates)
        
        if self.nonlinearity == "tanh":
            h = torch.tanh(gates)
        else:
            h = torch.relu(gates)
            
        return h


class SimpleRNN(BaseModel):
    def __init__(
        self, 
        input_size=13,
        hidden_size=128,
        task_embedding_dim=32,
        num_tasks=34,
        dropout=0.3,
        embedding_dropout=0.1,
        zoneout_prob=0.1,
        activity_l1=0.01,
        verbose=False
    ):
        super().__init__()
        
        self.save_hyperparameters()
        self.verbose = verbose
        
        # Feature-wise layer normalization
        self.feature_norm = nn.LayerNorm(input_size)
        
        # Task embedding with dropout
        self.task_embedding = nn.Embedding(num_tasks + 1, task_embedding_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        
        # Combined input size
        rnn_input_size = input_size + task_embedding_dim
        
        # Custom RNN cell with layer normalization
        self.rnn_cell = LayerNormRNNCell(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            nonlinearity="tanh"
        )
        
        # Variational dropout
        self.variational_dropout = VariationalDropout(dropout)
        self.zoneout_prob = zoneout_prob
        self.activity_l1 = activity_l1
        
        # Output layer
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        self._init_weights()
                
    def set_class_weights(self, dataset):
        if hasattr(dataset, 'class_weights'):
            self.class_weights = torch.tensor([
                dataset.class_weights[0],
                dataset.class_weights[1]
            ], device=self.device)

    def _init_weights(self):
        """Initialize weights with scaled orthogonal initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and 'norm' not in name and 'embedding' not in name:
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
        # Initialize embedding with smaller values
        nn.init.uniform_(self.task_embedding.weight, -0.05, 0.05)

    def forward(self, x, task_ids, masks=None):
        batch_size, seq_len, features = x.size()
        
        # Initial preprocessing
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        x = torch.clamp(x, -10, 10)
        
        # Apply layer normalization to each time step independently
        x = self.feature_norm(x)
        
        # Process task embeddings
        task_ids = task_ids.squeeze(-1)
        task_emb = self.task_embedding(task_ids)
        task_emb = self.embedding_dropout(task_emb)
        task_emb = task_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine features with task embeddings
        x = torch.cat([x, task_emb], dim=-1)
        
        # Apply variational dropout
        x = self.variational_dropout(x)
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.rnn_cell.hidden_size, device=x.device)
        hidden_states = []
        
        # RNN forward pass with zoneout
        for t in range(seq_len):
            h_prev = h.clone()
            h = self.rnn_cell(x[:, t, :], h)
            
            # Apply zoneout
            if self.training and self.zoneout_prob > 0:
                mask = torch.bernoulli(
                    torch.full_like(h, self.zoneout_prob)
                )
                h = mask * h_prev + (1 - mask) * h
            
            hidden_states.append(h)
        
        # Stack hidden states
        hidden_states = torch.stack(hidden_states, dim=1)
        
        # Apply masking if provided
        if masks is not None:
            hidden_states = hidden_states * masks.unsqueeze(-1)
        
        # Add activity regularization
        if self.training and self.activity_l1 > 0:
            self.activity_regularization = self.activity_l1 * torch.mean(torch.abs(hidden_states))
        else:
            self.activity_regularization = 0
        
        # Use final hidden state for classification
        return self.classifier(hidden_states[:, -1, :])

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        if hasattr(self, 'activity_regularization'):
            loss = loss + self.activity_regularization
        return loss

    def configure_optimizers(self):
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