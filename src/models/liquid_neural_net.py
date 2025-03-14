import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models.base import BaseModel


class SimpleLiquidCell(nn.Module):
    """Simplified liquid time-constant cell for efficient processing.
    
    This cell implements a streamlined version of the liquid time-constant
    computation that is numerically stable and computationally efficient.
    """
    def __init__(self, input_size, hidden_size, dropout=0.1):
        """Initialize the simplified liquid cell.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
            dropout: Dropout probability
        """
        super().__init__()
        
        # Core transformation matrices
        self.weight_ih = nn.Linear(input_size, hidden_size)
        self.weight_hh = nn.Linear(hidden_size, hidden_size)
        
        # Adaptive time constant as a simpler function
        self.tau_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Normalization and regularization
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters for stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability."""
        # Initialize linear layers with safe methods
        nn.init.xavier_uniform_(self.weight_ih.weight)
        nn.init.xavier_uniform_(self.weight_hh.weight)
        nn.init.zeros_(self.weight_ih.bias)
        nn.init.zeros_(self.weight_hh.bias)
        
        # Initialize tau gate weights carefully
        for name, param in self.tau_gate.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x, h_prev, dt=0.1):
        """Forward pass with adaptive dynamics.
        
        Args:
            x: Input tensor
            h_prev: Previous hidden state
            dt: Base timestep size
            
        Returns:
            Updated hidden state
        """
        # Compute input projection with dropout
        x_proj = self.dropout(self.weight_ih(x))
        
        # Compute recurrent projection
        h_proj = self.weight_hh(h_prev)
        
        # Compute target hidden state
        h_target = self.layer_norm(x_proj + h_proj)
        
        # Compute adaptive time constant (bounded between 0.2 and 2.0)
        tau = 0.2 + 1.8 * self.tau_gate(h_target)
        
        # Update hidden state using liquid time-constant formula
        h_new = h_prev + (dt / tau) * (h_target - h_prev)
        
        return h_new


class LiquidNetwork(BaseModel):
    """Liquid Neural Network for handwriting analysis.
    
    This model implements a Liquid Neural Networks that focuses on computational 
    efficiency while preserving the adaptive
    time-constant dynamics that make LNNs valuable for sequential data.
    """
    def __init__(
        self,
        input_size=13,
        hidden_size=128,
        num_layers=2,
        num_tasks=34,
        task_embedding_dim=32,
        dropout=0.3,
        dt=0.1,
        bidirectional=False,
        verbose=False
    ):
        """Initialize the simplified liquid neural network.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
            num_layers: Number of liquid layers
            num_tasks: Number of different tasks
            task_embedding_dim: Dimension of task embeddings
            dropout: Dropout probability
            dt: Base timestep size
            bidirectional: Whether to use bidirectional processing
            verbose: Whether to print debug information
        """
        super().__init__()
        
        self.save_hyperparameters()
        self.verbose = verbose
        self.dt = dt
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(input_size)
        
        # Task embedding
        self.task_embedding = nn.Embedding(num_tasks + 1, task_embedding_dim)
        
        # Task-specific projection
        self.task_proj = nn.Linear(task_embedding_dim, hidden_size)
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        
        # Create liquid cells for each direction and layer
        self.forward_cells = nn.ModuleList([
            SimpleLiquidCell(
                input_size=hidden_size,
                hidden_size=hidden_size,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        if bidirectional:
            self.backward_cells = nn.ModuleList([
                SimpleLiquidCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ])
            
            # Output size accounting for bidirectionality
            output_size = hidden_size * 2
        else:
            output_size = hidden_size
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Dropout(dropout),
            nn.Linear(output_size, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with safe initialization methods."""
        # Initialize embeddings
        nn.init.normal_(self.task_embedding.weight, mean=0.0, std=0.01)
        
        # Initialize linear layers
        for layer in [self.task_proj, self.input_proj]:
            if hasattr(layer, 'weight') and layer.weight.dim() > 1:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Initialize classifier with safe methods
        for name, param in self.classifier.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def _process_sequence(self, x, task_mod, cells, h_init, forward=True):
        """Process sequence in forward or backward direction.
        
        Args:
            x: Input sequence (batch_size, seq_len, hidden_size)
            task_mod: Task modulation (batch_size, hidden_size)
            cells: List of liquid cells for this direction
            h_init: Initial hidden states
            forward: Whether to process in forward direction
            
        Returns:
            Final hidden state from last layer
        """
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden states for each layer
        h_states = [h_init.clone() for _ in range(self.num_layers)]
        
        # Determine sequence processing order
        timesteps = range(seq_len) if forward else range(seq_len - 1, -1, -1)
        
        # Process sequence
        for t in timesteps:
            x_t = x[:, t]  # (batch_size, hidden_size)
            
            # Add task modulation
            x_t = x_t + task_mod
            
            # Process through layers
            for layer_idx, cell in enumerate(cells):
                # Update hidden state
                h_states[layer_idx] = cell(x_t, h_states[layer_idx], self.dt)
                
                # Update input for next layer
                x_t = h_states[layer_idx]
        
        # Return final hidden state from last layer
        return h_states[-1]
    
    def forward(self, x, task_ids, masks=None):
        """Forward pass of the simplified liquid neural network.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            task_ids: Task identifiers (batch_size, 1)
            masks: Optional mask tensor (batch_size, seq_len)
            
        Returns:
            Logits for classification
        """
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        # Initial preprocessing
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        x = torch.clamp(x, -10, 10)
        
        # Apply feature normalization
        x = self.feature_norm(x)
        
        # Process task embeddings
        task_ids = task_ids.squeeze(-1)
        task_emb = self.task_embedding(task_ids)
        task_emb = self.dropout(task_emb)
        
        # Project tasks to hidden dimension
        task_mod = self.task_proj(task_emb)
        
        # Project input features
        x = self.input_proj(x)
        
        # Apply masking if provided
        if masks is not None:
            x = x * masks.unsqueeze(-1)
        
        # Initialize hidden states
        h_init = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # Process sequence in forward direction
        forward_output = self._process_sequence(
            x, task_mod, self.forward_cells, h_init, forward=True
        )
        
        # Process sequence in backward direction if bidirectional
        if self.bidirectional:
            backward_output = self._process_sequence(
                x, task_mod, self.backward_cells, h_init, forward=False
            )
            
            # Concatenate both directions
            final_output = torch.cat([forward_output, backward_output], dim=-1)
        else:
            final_output = forward_output
        
        # Apply classifier
        logits = self.classifier(final_output)
        
        return logits
    
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
        
        # Calculate weighted average based on prediction confidence
        final_preds = {}
        for subj, preds in subject_preds.items():
            preds_array = np.array(preds)
            confidence = np.abs(preds_array - 0.5) + 0.5
            weighted_avg = np.average(preds_array, weights=confidence)
            final_preds[subj] = weighted_avg > 0.5
            
        return final_preds