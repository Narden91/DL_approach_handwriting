import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print as rprint

from src.models.base import BaseModel
from src.models.RNN import RNNDebugger


class ResidualConnection(nn.Module):
    """Simple residual connection for improved gradient flow.
    
    This replaces the more complex HighwayLayer with a simpler
    residual connection that still facilitates gradient flow.
    """
    def __init__(self, size, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, residual):
        """Apply residual connection with normalization.
        
        Args:
            x: Main path tensor
            residual: Residual path tensor
            
        Returns:
            Combined tensor with residual connection
        """
        return self.norm(x + self.dropout(residual))


class XLSTM(BaseModel):
    """Extended LSTM model optimized for handwriting analysis.
    
    This model implements a simplified version of XLSTM that reduces
    architectural complexity while focusing on the core strengths
    needed for handwriting sequence analysis.
    """
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
        bidirectional=True,
        verbose=False
    ):
        """Initialize the streamlined XLSTM model.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of LSTM hidden states
            num_layers: Number of LSTM layers
            num_tasks: Number of different tasks
            task_embedding_dim: Dimension of task embeddings
            dropout: General dropout rate
            embedding_dropout: Dropout rate for embeddings
            layer_norm: Whether to use layer normalization
            recurrent_dropout: Dropout rate for recurrent connections
            bidirectional: Whether to use bidirectional LSTM
            verbose: Whether to print debug information
        """
        super().__init__()
        
        self.save_hyperparameters()
        self.verbose = verbose
        self.debugger = RNNDebugger()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(input_size)
        
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
            bidirectional=bidirectional
        )
        
        # Output size considering bidirectional
        output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Simple residual connection
        self.residual = ResidualConnection(output_size, dropout)
        
        # Classifier with layer normalization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_size, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with appropriate initialization schemes."""
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
                nn.init.xavier_uniform_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def set_class_weights(self, dataset):
        """Set class weights from dataset for loss computation."""
        if hasattr(dataset, 'class_weights'):
            self.class_weights = torch.tensor([
                dataset.class_weights[0],
                dataset.class_weights[1]
            ], device=self.device)

    def forward(self, x, task_ids, masks=None):
        """Forward pass of the streamlined XLSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            task_ids: Task identifiers of shape (batch_size, 1)
            masks: Optional mask tensor of shape (batch_size, seq_len)
            
        Returns:
            Logits for classification
        """
        batch_size, seq_len, _ = x.size()
        
        # Initial preprocessing with adaptive clipping
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        # Adaptive clipping based on distribution stats
        with torch.no_grad():
            mean = x.mean()
            std = x.std()
            clip_min = mean - 5 * std
            clip_max = mean + 5 * std
        
        x = torch.clamp(x, clip_min, clip_max)
        
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
        
        # Initialize hidden and cell states
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=x.device
        )
        c0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=x.device
        )
        
        # LSTM forward pass
        # Store original input for residual connection
        original_x = x
        output, (h_n, _) = self.lstm(x, (h0, c0))
        
        if self.verbose:
            self.debugger.check_tensor(output, "LSTM output", "Forward")
        
        # Process the last hidden state from each direction if bidirectional
        if self.bidirectional:
            # Reshape to separate layers and directions
            h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_size)
            # Take last layer's hidden states from both directions
            h_forward = h_n[-1, 0]  # Last layer, forward direction
            h_backward = h_n[-1, 1]  # Last layer, backward direction
            # Concatenate both directions
            final_hidden = torch.cat([h_forward, h_backward], dim=-1)
        else:
            # Just take the last layer's hidden state
            final_hidden = h_n[-1]
        
        # Apply residual connection if shapes are compatible
        if seq_len > 0 and original_x.size(-1) == output.size(-1):
            # Use the mean of original sequence for the residual
            residual_input = original_x.mean(dim=1)
            if residual_input.size(-1) == final_hidden.size(-1):
                final_output = self.residual(final_hidden, residual_input)
            else:
                final_output = final_hidden
        else:
            final_output = final_hidden
            
        return self.classifier(final_output)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.model_config['learning_rate'],
            weight_decay=self.model_config['weight_decay'],
            amsgrad=True
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
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
        """Aggregate window-level predictions to subject-level using confidence weighting."""
        pred_probs = torch.sigmoid(window_preds).cpu().numpy()
        subject_preds = {}
        
        for pred, subj_id in zip(pred_probs, subject_ids):
            if subj_id not in subject_preds:
                subject_preds[subj_id] = []
            subject_preds[subj_id].append(pred)
        
        # Calculate weighted average based on prediction confidence
        final_preds = {}
        for subj, preds in subject_preds.items():
            preds_array = np.array(preds)
            # Higher weight for predictions further from decision boundary
            confidence = np.abs(preds_array - 0.5) + 0.5
            weighted_avg = np.average(preds_array, weights=confidence)
            final_preds[subj] = weighted_avg > 0.5
            
        return final_preds