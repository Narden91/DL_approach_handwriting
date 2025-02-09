import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print as rprint

from src.models.base import BaseModel
from src.models.RNN import RNNDebugger


class HighwayLayer(nn.Module):
    """Highway Connection Layer for LSTMs."""
    def __init__(self, size):
        super().__init__()
        self.transform_gate = nn.Linear(size, size)
        self.carry_gate = nn.Linear(size, size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.transform_gate.weight)
        nn.init.xavier_uniform_(self.carry_gate.weight)
        nn.init.zeros_(self.transform_gate.bias)
        nn.init.ones_(self.carry_gate.bias)

    def forward(self, x, h):
        transform = torch.sigmoid(self.transform_gate(h))
        carry = torch.sigmoid(self.carry_gate(h))
        return transform * x + carry * h
    

class XLSTM(BaseModel):
    def __init__(
        self, 
        input_size=13,
        hidden_size=128,
        num_layers=2,
        num_tasks=34,
        task_embedding_dim=32,
        dropout=0.3,
        embedding_dropout=0.1,
        layer_norm=True,
        recurrent_dropout=0.1,
        verbose=False
    ):
        super().__init__()
        
        self.save_hyperparameters()
        self.verbose = verbose
        self.debugger = RNNDebugger()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Feature normalization layers
        self.feature_norm = nn.LayerNorm(input_size)
        self.input_norm = nn.BatchNorm1d(input_size, eps=1e-5, momentum=0.1)
        
        # Task embedding with dropout
        self.task_embedding = nn.Embedding(num_tasks + 1, task_embedding_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        
        # Combined input size
        lstm_input_size = input_size + task_embedding_dim
        
        # Main LSTM layer
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Highway layers for improved information flow
        self.highway = HighwayLayer(hidden_size * 2)
        
        # Output size considering bidirectional
        output_size = hidden_size * 2
        
        # Classifier with layer normalization
        self.classifier = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Dropout(dropout),
            nn.Linear(output_size, 1)
        )
        
        self._init_weights()
        
        if verbose:
            rprint(f"\n[bold blue]XLSTM Model Configuration:[/bold blue]")
            rprint(f"Input Size: {input_size}")
            rprint(f"Hidden Size: {hidden_size}")
            rprint(f"Number of Layers: {num_layers}")
            rprint(f"Task Embedding Dim: {task_embedding_dim}")
            rprint(f"Dropout: {dropout}")
            rprint(f"Embedding Dropout: {embedding_dropout}")
            rprint(f"Recurrent Dropout: {recurrent_dropout}")
            rprint(f"Layer Norm: {layer_norm}")
        
        # # Output size considering bidirectional
        # output_size = hidden_size * 2
        
        # # Classifier with layer normalization
        # self.classifier = nn.Sequential(
        #     nn.LayerNorm(output_size),
        #     nn.Dropout(dropout),
        #     nn.Linear(output_size, 1)
        # )
        
        # self._init_weights()
        
        # if verbose:
        #     rprint(f"\n[bold blue]XLSTM Model Configuration:[/bold blue]")
        #     rprint(f"Input Size: {input_size}")
        #     rprint(f"Hidden Size: {hidden_size}")
        #     rprint(f"Number of Layers: {num_layers}")
        #     rprint(f"Task Embedding Dim: {task_embedding_dim}")
        #     rprint(f"Dropout: {dropout}")
        #     rprint(f"Embedding Dropout: {embedding_dropout}")
        #     rprint(f"Recurrent Dropout: {recurrent_dropout}")
        #     rprint(f"Layer Norm: {layer_norm}")

    def _init_weights(self):
        """Initialize weights with appropriate initialization schemes"""
        # Initialize embedding with smaller values
        nn.init.uniform_(self.task_embedding.weight, -0.05, 0.05)
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param, gain=1.0)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.)
        
        # Initialize classifier
        for name, param in self.classifier.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param, gain=0.1)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def set_class_weights(self, dataset):
        """Set class weights from dataset"""
        if hasattr(dataset, 'class_weights'):
            self.class_weights = torch.tensor([
                dataset.class_weights[0],
                dataset.class_weights[1]
            ], device=self.device)

    def forward(self, x, task_ids, masks=None):
        """
        Forward pass of the model.
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            task_ids: Task identifiers of shape (batch_size, 1)
            masks: Optional mask tensor of shape (batch_size, seq_len)
        """
        batch_size, seq_len, _ = x.size()
        
        # Initial preprocessing
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        x = torch.clamp(x, -10, 10)
        
        if self.verbose:
            self.debugger.check_tensor(x, "Initial input", "Forward")
        
        # Apply normalization
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.feature_norm(x)
        
        # Process task embeddings
        task_ids = task_ids.squeeze(-1)
        task_emb = self.task_embedding(task_ids)
        task_emb = self.embedding_dropout(task_emb)
        task_emb = task_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine features with task embeddings
        x = torch.cat([x, task_emb], dim=-1)
        
        if self.verbose:
            self.debugger.check_tensor(x, "Combined input", "Forward")
        
        # Apply masking if provided
        if masks is not None:
            x = x * masks.unsqueeze(-1)
        
        output, (h_n, _) = self.lstm(x)

        # Fixing shape mismatch in Highway Connection
        h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_size)
        h_n = torch.cat((h_n[-1, 0], h_n[-1, 1]), dim=-1)  # Concatenating both directions

        # Apply Highway Connection
        output = self.highway(output[:, -1, :], h_n) # Ensuring correct shape
        output = output.view(batch_size, -1)  # Flatten for classifier
        return self.classifier(output)

    
        # batch_size, seq_len, _ = x.size()
        
        # # Initial preprocessing
        # x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        # x = torch.clamp(x, -10, 10)
        
        # if self.verbose:
        #     self.debugger.check_tensor(x, "Initial input", "Forward")
        
        # # Apply normalization
        # x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        # x = self.feature_norm(x)
        
        # # Process task embeddings
        # task_ids = task_ids.squeeze(-1)
        # task_emb = self.task_embedding(task_ids)
        # task_emb = self.embedding_dropout(task_emb)
        # task_emb = task_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # # Combine features with task embeddings
        # x = torch.cat([x, task_emb], dim=-1)
        
        # if self.verbose:
        #     self.debugger.check_tensor(x, "Combined input", "Forward")
        
        # # Apply masking if provided
        # if masks is not None:
        #     x = x * masks.unsqueeze(-1)
        
        # # LSTM forward pass
        # output, _ = self.lstm(x)
        
        # if self.verbose:
        #     self.debugger.check_tensor(output, "LSTM output", "Forward")
        
        # # Use final output for classification
        # return self.classifier(output[:, -1, :])
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
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