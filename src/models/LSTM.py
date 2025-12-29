import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print as rprint

from src.models.base import BaseModel


class LSTMDebugger:
    """Debug utility for LSTM models with comprehensive tensor and gradient checking.
    
    This class provides static methods to check tensors for numerical anomalies
    and monitor gradient flow during training, which helps identify and diagnose
    training issues early.
    """
    @staticmethod
    def check_tensor(tensor: torch.Tensor, name: str, step: str) -> bool:
        """Check tensor for NaN/Inf values and print statistics.
        
        Args:
            tensor: Tensor to check
            name: Descriptive name for the tensor
            step: Current processing step for context
            
        Returns:
            True if tensor is valid, False otherwise
        """
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
    def check_gradients(model: nn.Module, step: str) -> bool:
        """Check gradients for NaN/Inf values and print statistics.
        
        Args:
            model: Model to check gradients for
            step: Current processing step for context
            
        Returns:
            True if all gradients are valid, False otherwise
        """
        all_valid = True
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                param_norm = param.norm()
                
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    rprint(f"[red]{step} - Gradient NaN/Inf detected in {name}[/red]")
                    all_valid = False
                    continue
                    
                # Only print for significant parameters
                if param.numel() > 100 or grad_norm > 1.0:
                    rprint(f"[blue]{step} - {name} - grad norm: {grad_norm:.4f}, param norm: {param_norm:.4f}[/blue]")
        
        return all_valid


class TemporalAttention(nn.Module):
    """Temporal attention mechanism to focus on relevant time steps in sequence data.
    
    This module implements a learnable attention mechanism that weighs the importance
    of different time steps in the sequence, allowing the model to focus on the most
    relevant parts of the handwriting for classification.
    
    Attributes:
        attention_weights: Learned projection for computing attention scores
        layer_norm: Layer normalization for stable training
    """
    def __init__(self, hidden_size, attention_dim=64, dropout=0.1):
        """Initialize the temporal attention mechanism.
        
        Args:
            hidden_size: Size of the hidden state from LSTM
            attention_dim: Dimension of the attention projection
            dropout: Dropout probability for regularization
        """
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states, mask=None):
        """Apply attention mechanism to sequence of hidden states.
        
        Args:
            hidden_states: Sequence of hidden states (batch_size, seq_len, hidden_size)
            mask: Optional mask for padding (batch_size, seq_len)
            
        Returns:
            context: Context vector summarizing the sequence (batch_size, hidden_size)
            attention: Attention weights (batch_size, seq_len)
        """
        # Project to get attention scores
        scores = self.projection(hidden_states).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention = F.softmax(scores, dim=1)
        attention = self.dropout(attention)
        
        # Apply attention weights to hidden states
        context = torch.bmm(attention.unsqueeze(1), hidden_states).squeeze(1)
        context = self.layer_norm(context)
        
        return context, attention


class LSTM(BaseModel):
    """LSTM model for handwriting sequence analysis.
    
    This model implements an LSTM architecture with optional bidirectionality,
    layer normalization, and temporal attention mechanisms. It is designed
    to process handwriting data sequences and classify samples as healthy
    or unhealthy.
    
    Attributes:
        feature_norm: Normalization layer for input features
        task_embedding: Embedding layer for task identifiers
        lstm: Main LSTM layer
        attention: Optional temporal attention mechanism
        classifier: Output classification layer
        verbose: Whether to print debug information
        debugger: Instance of LSTMDebugger for debugging
    """
    def __init__(
        self, 
        input_size=13,
        hidden_size=128,
        num_layers=2,
        task_embedding_dim=32,
        num_tasks=34,
        dropout=0.3,
        embedding_dropout=0.1,
        layer_norm=True,
        bidirectional=True,
        use_attention=True,
        attention_dim=64,
        verbose=False
    ):
        """Initialize the LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            num_layers: Number of LSTM layers
            task_embedding_dim: Dimension of task embeddings
            num_tasks: Number of unique tasks
            dropout: Dropout probability
            embedding_dropout: Dropout probability for embeddings
            layer_norm: Whether to use layer normalization
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use temporal attention
            attention_dim: Dimension of attention projection
            verbose: Whether to print debug information
        """
        super().__init__()
        
        self.save_hyperparameters()
        self.verbose = verbose
        self.debugger = LSTMDebugger()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(input_size)
        
        # Task embedding
        self.task_embedding = nn.Embedding(num_tasks + 1, task_embedding_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        
        # Combined input size
        lstm_input_size = input_size + task_embedding_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output size considering bidirectionality
        output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention mechanism
        if use_attention:
            self.attention = TemporalAttention(
                hidden_size=output_size,
                attention_dim=attention_dim,
                dropout=dropout
            )
        
        # Output classifier with layer normalization
        if layer_norm:
            self.classifier = nn.Sequential(
                nn.LayerNorm(output_size),
                nn.Dropout(dropout),
                nn.Linear(output_size, 1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(output_size, 1)
            )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with appropriate initialization schemes for LSTMs.
        
        Uses orthogonal initialization for recurrent weights and Xavier uniform
        for input-to-hidden weights and linear layers.
        """
        # Initialize embedding
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
                param.data[n//4:n//2].fill_(1.)  # Set forget gate bias to 1
        
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
        """Set class weights from dataset for weighted loss calculation.
        
        Args:
            dataset: Dataset instance containing class weights
        """
        if hasattr(dataset, 'class_weights'):
            self.class_weights = torch.tensor([
                dataset.class_weights[0],
                dataset.class_weights[1]
            ], device=self.device)
                
    def forward(self, x, task_ids, masks=None):
        """Forward pass of the LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            task_ids: Task identifier tensor of shape (batch_size, 1)
            masks: Optional mask tensor of shape (batch_size, seq_len)
            
        Returns:
            Model output logits of shape (batch_size, 1)
        """
        batch_size, seq_len, _ = x.size()
        
        # Initial preprocessing with adaptive clipping
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        # Dynamic clipping based on distribution
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
        
        # Initialize hidden state and cell state
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
        output, _ = self.lstm(x, (h0, c0))
        
        if self.verbose:
            self.debugger.check_tensor(output, "LSTM output", "Forward")
        
        # Apply attention if enabled
        if self.use_attention:
            context, attention_weights = self.attention(output, masks)
            
            # Store attention weights for analysis if in training mode
            if self.training:
                self.attention_weights = attention_weights
                
            final_output = context
        else:
            # Use final output for classification
            final_output = output[:, -1, :]
        
        return self.classifier(final_output)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler.
        
        Returns:
            Dictionary containing optimizer and scheduler configuration
        """
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
            min_lr=1e-6
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
        """Aggregate window-level predictions to subject-level with confidence weighting.
        
        Args:
            window_preds: Prediction tensor for windows
            subject_ids: List of subject IDs corresponding to predictions
            
        Returns:
            Dictionary mapping subject IDs to aggregated predictions
        """
        pred_probs = torch.sigmoid(window_preds).cpu().numpy()
        subject_preds = {}
        
        for pred, subj_id in zip(pred_probs, subject_ids):
            if subj_id not in subject_preds:
                subject_preds[subj_id] = []
            subject_preds[subj_id].append(pred)
        
        # Use confidence-weighted average for better aggregation
        final_preds = {}
        for subj, preds in subject_preds.items():
            preds_array = np.array(preds)
            
            # Higher confidence for predictions farther from decision boundary
            confidence = np.abs(preds_array - 0.5) * 2  # Scale to [0, 1]
            
            # Avoid division by zero
            if np.sum(confidence) > 0:
                weighted_avg = np.sum(preds_array * confidence) / np.sum(confidence)
            else:
                weighted_avg = np.mean(preds_array)
                
            final_preds[subj] = weighted_avg > 0.5
            
        return final_preds