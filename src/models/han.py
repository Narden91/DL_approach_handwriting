import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rich import print as rprint

from src.models.base import BaseModel


class AttentionLayer(nn.Module):
    """Attention mechanism for focusing on relevant features.
    
    This module implements a flexible attention mechanism that can be applied
    at different levels of the hierarchy (stroke, feature, task).
    """
    def __init__(self, input_dim, attention_dim=64, dropout=0.1, use_layer_norm=True):
        """Initialize attention layer.
        
        Args:
            input_dim: Dimension of input features
            attention_dim: Dimension of attention projection
            dropout: Dropout probability
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.use_layer_norm = use_layer_norm
        
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim)
    
    def forward(self, inputs, mask=None):
        """Apply attention mechanism to inputs.
        
        Args:
            inputs: Input sequence (batch_size, seq_len, input_dim)
            mask: Optional mask for padding (batch_size, seq_len)
            
        Returns:
            Tuple of (context vector, attention weights)
        """
        # Compute attention scores
        scores = self.attention(inputs).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        weights = F.softmax(scores, dim=1)
        weights = self.dropout(weights)
        
        # Apply attention weights to inputs
        context = torch.bmm(weights.unsqueeze(1), inputs).squeeze(1)
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            context = self.layer_norm(context)
            
        return context, weights


class TaskEncoder(nn.Module):
    """Encoder for task-specific information.
    
    This module processes task identifiers and produces task embeddings
    that can be used to modulate other features.
    """
    def __init__(self, num_tasks, embedding_dim, hidden_dim, dropout=0.1):
        """Initialize task encoder.
        
        Args:
            num_tasks: Number of unique tasks
            embedding_dim: Initial embedding dimension
            hidden_dim: Dimension of task representation
            dropout: Dropout probability
        """
        super().__init__()
        
        self.task_embedding = nn.Embedding(num_tasks + 1, embedding_dim)
        self.task_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, task_ids):
        """Encode task identifiers.
        
        Args:
            task_ids: Task identifier tensor (batch_size, 1)
            
        Returns:
            Task embeddings (batch_size, hidden_dim)
        """
        task_ids = task_ids.squeeze(-1)
        task_emb = self.task_embedding(task_ids)
        return self.task_projection(task_emb)


class HandwritingHAN(BaseModel):
    """Hierarchical Attention Network for handwriting analysis.
    
    This model implements a hierarchical attention architecture specifically
    designed for handwriting data, with separate processing for static and
    dynamic features and multiple levels of attention.
    """
    def __init__(
        self,
        feature_dims,  # Dictionary with 'static' and 'dynamic' dimensions
        hidden_size=128,
        attention_dim=64,
        num_tasks=34,
        task_embedding_dim=32,
        dropout=0.3,
        feature_dropout=0.2,
        attention_dropout=0.1,
        use_layer_norm=True,
        verbose=False
    ):
        """Initialize HandwritingHAN model.
        
        Args:
            feature_dims: Dictionary with 'static' and 'dynamic' feature dimensions
            hidden_size: Size of hidden representations
            attention_dim: Dimension of attention mechanism
            num_tasks: Number of unique tasks
            task_embedding_dim: Dimension of task embeddings
            dropout: General dropout probability
            feature_dropout: Dropout for feature encoders
            attention_dropout: Dropout for attention mechanisms
            use_layer_norm: Whether to use layer normalization
            verbose: Whether to print debug information
        """
        super().__init__()
        
        self.save_hyperparameters(ignore=['feature_dims'])
        self.verbose = verbose
        
        # Store feature dimensions
        self.static_dim = feature_dims['static']
        self.dynamic_dim = feature_dims['dynamic']
        
        # Task encoder
        self.task_encoder = TaskEncoder(
            num_tasks=num_tasks,
            embedding_dim=task_embedding_dim,
            hidden_dim=hidden_size,
            dropout=dropout
        )
        
        # Feature normalization
        self.static_norm = nn.LayerNorm(self.static_dim)
        self.dynamic_norm = nn.LayerNorm(self.dynamic_dim)
        
        # Feature encoders
        self.static_encoder = nn.Sequential(
            nn.Linear(self.static_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(feature_dropout)
        )
        
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(self.dynamic_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(feature_dropout)
        )
        
        # Attention mechanisms
        self.feature_attention = AttentionLayer(
            input_dim=hidden_size,
            attention_dim=attention_dim,
            dropout=attention_dropout,
            use_layer_norm=use_layer_norm
        )
        
        # Multimodal fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 2 + hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final classifier
        self.classifier = nn.Linear(hidden_size, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability."""
        # Initialize task embedding
        nn.init.uniform_(self.task_encoder.task_embedding.weight, -0.05, 0.05)
        
        # Initialize linear layers
        for module in [self.static_encoder, self.dynamic_encoder, 
                      self.fusion_layer, self.feature_attention]:
            for name, param in module.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    nn.init.xavier_uniform_(param, gain=0.1)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight, gain=0.01)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x, task_ids, masks=None):
        """Forward pass of the HandwritingHAN model.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            task_ids: Task identifiers (batch_size, 1)
            masks: Optional mask tensor (batch_size, seq_len)
            
        Returns:
            Classification logits (batch_size, 1)
        """
        batch_size = x.size(0)
        device = x.device
        
        # Split features into static and dynamic
        static_x = x[:, 0, :self.static_dim]  # Take first timestep for static features
        dynamic_x = x[:, :, self.static_dim:]  # All timesteps for dynamic features
        
        if self.verbose:
            rprint(f"[blue]Static features shape: {static_x.shape}[/blue]")
            rprint(f"[blue]Dynamic features shape: {dynamic_x.shape}[/blue]")
        
        # Normalize features
        static_x = self.static_norm(static_x)
        dynamic_x = self.dynamic_norm(dynamic_x)
        
        # Encode features
        static_encoded = self.static_encoder(static_x)  # (batch_size, hidden_size)
        dynamic_encoded = self.dynamic_encoder(dynamic_x)  # (batch_size, seq_len, hidden_size)
        
        # Encode task
        task_encoded = self.task_encoder(task_ids)  # (batch_size, hidden_size)
        
        # Apply feature-level attention to dynamic features
        dynamic_context, attention_weights = self.feature_attention(dynamic_encoded, masks)
        
        # Store attention weights for analysis
        if self.training:
            self.last_attention_weights = attention_weights
        
        # Combine static features, dynamic context, and task embedding
        combined = torch.cat([static_encoded, dynamic_context, task_encoded], dim=1)
        
        # Apply fusion layer
        fused_representation = self.fusion_layer(combined)
        
        # Final classification
        logits = self.classifier(fused_representation)
        
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
    
    def get_attention_weights(self):
        """Get attention weights for interpretation.
        
        Returns:
            Dictionary of attention weights from last forward pass
        """
        if not hasattr(self, 'last_attention_weights'):
            return None
        
        return self.last_attention_weights.detach().cpu().numpy()
    
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