import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print as rprint

from src.models.base import BaseModel
from src.models.RNN import RNNDebugger


class GRU(BaseModel):
    def __init__(
        self, 
        input_size=13,
        hidden_size=128,
        num_layers=2,
        num_tasks=34,
        task_embedding_dim=32,
        dropout=0.1,
        batch_first=True,
        bidirectional=True,
        bias=True,
        verbose=False
    ):
        super().__init__()
        
        self.save_hyperparameters()
        self.verbose = verbose
        self.debugger = RNNDebugger()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(input_size)
        
        # Task embedding
        self.task_embedding = nn.Embedding(num_tasks + 1, task_embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Combined input size
        rnn_input_size = input_size + task_embedding_dim
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            bias=bias
        )
        
        # Output size considering bidirectional
        output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Output classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Dropout(dropout),
            nn.Linear(output_size, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with appropriate initialization"""
        # Initialize embedding
        nn.init.uniform_(self.task_embedding.weight, -0.05, 0.05)
        
        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param, gain=1.0)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1
                nn.init.constant_(param[self.hidden_size:2*self.hidden_size], 1.0)
        
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
        
        # Apply layer normalization
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
        
        # Initialize hidden state
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=x.device
        )
        
        # GRU forward pass
        output, _ = self.gru(x, h0)
        
        if self.verbose:
            self.debugger.check_tensor(output, "GRU output", "Forward")
        
        # Use final output for classification
        final_output = output[:, -1, :]
        
        return self.classifier(final_output)
    
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