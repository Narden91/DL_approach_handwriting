import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print as rprint

from src.models.base import BaseModel


class RNNDebugger:
    """Utility class for debugging RNN models.
    
    This class provides static methods to check tensors for anomalies
    and monitor gradients, helping with debugging training issues.
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
            
        # Calculate statistics only if in verbose mode to save computation
        if hasattr(tensor, 'device'):
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
    """Recurrent Neural Network model for handwriting analysis.
    
    This model implements a standard RNN architecture with optional
    bidirectionality, layer normalization, and residual connections.
    
    Attributes:
        input_norm: Normalization layer for input features
        task_embedding: Embedding layer for task identifiers
        rnn: Main RNN layer
        classifier: Output classification layer
        verbose: Whether to print debug information
        debugger: Instance of RNNDebugger for debugging
    """
    def __init__(
        self, 
        input_size=13,
        hidden_size=128, 
        num_layers=2,
        num_tasks=34, 
        task_embedding_dim=32, 
        nonlinearity="tanh",
        dropout=0.1,
        bidirectional=True,
        layer_norm=True,
        residual=True,
        verbose=False
    ):
        """Initialize the RNN model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            num_layers: Number of RNN layers
            num_tasks: Number of unique tasks
            task_embedding_dim: Dimension of task embeddings
            nonlinearity: RNN activation function ('tanh' or 'relu')
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional RNN
            layer_norm: Whether to use layer normalization
            residual: Whether to use residual connections
            verbose: Whether to print debug information
        """
        super().__init__()
        
        self.save_hyperparameters()
        self.verbose = verbose
        self.bidirectional = bidirectional
        self.layer_norm = layer_norm
        self.residual = residual
        self.debugger = RNNDebugger()
        
        # Layer normalization (better for RNNs than BatchNorm)
        if layer_norm:
            self.feature_norm = nn.LayerNorm(input_size)
        else:
            # Fallback to BatchNorm for compatibility
            self.input_norm = nn.BatchNorm1d(input_size)
        
        # Task embedding (task IDs are 1-indexed, so we subtract 1 during forward)
        self.task_embedding = nn.Embedding(num_tasks, task_embedding_dim)
        
        # Combine feature dimensions with task embeddings
        rnn_input_size = input_size + task_embedding_dim
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output size considering bidirectionality
        output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Residual projection to match dimensions if residual connections enabled
        if residual:
            self.residual_projection = nn.Linear(rnn_input_size, output_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output classifier
        self.classifier = nn.Linear(output_size, 1)
        
        # Initialize weights with optimized strategy
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with orthogonal initialization for RNNs and
        Xavier uniform for linear layers."""
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param, gain=1.0)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
        # Initialize task embedding
        nn.init.uniform_(self.task_embedding.weight, -0.05, 0.05)
                
        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight, gain=0.1)
        nn.init.zeros_(self.classifier.bias)
                
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
        """Forward pass of the RNN model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            task_ids: Task identifier tensor of shape (batch_size, 1)
            masks: Optional mask tensor of shape (batch_size, seq_len)
            
        Returns:
            Model output logits of shape (batch_size, 1)
        """
        # Get tensor dimensions
        batch_size, seq_len, feature_dim = x.size()
        
        # Preprocessing: handle NaN/Inf, then clip all values
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = torch.clamp(x, -10, 10)
        
        # Apply normalization
        if self.layer_norm:
            x = self.feature_norm(x)
        else:
            # BatchNorm requires reshape: (B, T, F) -> (B*T, F) -> (B, T, F)
            x = self.input_norm(x.reshape(-1, feature_dim)).reshape(batch_size, seq_len, feature_dim)
        
        # Process task embeddings (task IDs are 1-indexed, adjust to 0-indexed)
        task_ids = task_ids.squeeze(-1)
        # Clamp to valid range and subtract 1 to convert from 1-indexed to 0-indexed
        task_ids = torch.clamp(task_ids, 1, self.task_embedding.num_embeddings) - 1
        task_emb = self.task_embedding(task_ids)
        task_emb = task_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine features with task embeddings
        x = torch.cat([x, task_emb], dim=-1)
        
        # Create identity connection for residual if enabled (after concatenation)
        if self.residual:
            identity = x
        
        if self.verbose:
            self.debugger.check_tensor(x, "Combined input", "Forward")
        
        # Apply masking if provided
        if masks is not None:
            x = x * masks.unsqueeze(-1)
        
        # RNN forward pass
        outputs, _ = self.rnn(x)
        
        if self.verbose:
            self.debugger.check_tensor(outputs, "RNN output", "Forward")
        
        # Apply residual connection if enabled
        if self.residual:
            # Project identity to match RNN output dimensions
            identity_projected = self.residual_projection(identity)
            outputs = outputs + identity_projected
        
        # Apply dropout
        outputs = self.dropout(outputs)
        
        # Use the final sequence output for classification
        final_outputs = outputs[:, -1, :]
        
        return self.classifier(final_outputs)

    def aggregate_predictions(self, window_preds, subject_ids):
        """Aggregate window-level predictions to subject-level.
        
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
        
        # Use weighted average based on prediction confidence
        final_preds = {}
        for subj, preds in subject_preds.items():
            preds_array = np.array(preds)
            # Calculate confidence weights - higher weight for predictions farther from decision boundary
            confidence = np.abs(preds_array - 0.5) + 0.5
            weighted_avg = np.average(preds_array, weights=confidence)
            final_preds[subj] = weighted_avg > 0.5
            
        return final_preds