import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models.base import BaseModel


class TemporalConvBlock(nn.Module):
    """Temporal convolution block with residual connections and dilation."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size-1)*dilation//2, dilation=dilation
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Optional residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        # x shape: (batch, channels, time)
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        return out + residual


class TemporalConvNet(nn.Module):
    """Temporal convolutional network with increasing dilation."""
    def __init__(self, in_channels, channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(channels)
        for i in range(num_levels):
            dilation = 2 ** i
            layers.append(TemporalConvBlock(
                in_channels if i == 0 else channels[i-1],
                channels[i], kernel_size, dilation, dropout
            ))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # x shape: (batch, time, channels) -> convert to (batch, channels, time)
        x = x.transpose(1, 2)
        x = self.network(x)
        # Convert back to (batch, time, channels)
        return x.transpose(1, 2)


class HierarchicalAttention(nn.Module):
    """Hierarchical attention mechanism for stroke, task, and subject levels."""
    def __init__(self, input_dim, attention_dim=64, dropout=0.1):
        super().__init__()
        
        # Stroke-level attention
        self.stroke_attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )
        
        # Task-level attention
        self.task_attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x, task_ids, masks=None):
        """Apply hierarchical attention.
        
        Args:
            x: Sequence features (batch_size, seq_len, input_dim)
            task_ids: Task identifiers (batch_size, 1)
            masks: Optional masks (batch_size, seq_len)
            
        Returns:
            Attended features and attention weights
        """
        batch_size, seq_len, feat_dim = x.size()
        
        # Apply stroke-level attention
        stroke_scores = self.stroke_attention(x).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply mask if provided
        if masks is not None:
            stroke_scores = stroke_scores.masked_fill(masks == 0, -1e9)
        
        # Normalize scores with softmax
        stroke_weights = F.softmax(stroke_scores, dim=1)
        stroke_weights = self.dropout(stroke_weights)
        
        # Apply attention weights to get task representations
        task_context = torch.bmm(stroke_weights.unsqueeze(1), x).squeeze(1)
        task_context = self.layer_norm(task_context)
        
        # Return both the attended features and attention weights for visualization
        return task_context, stroke_weights


class TaskEncoder(nn.Module):
    """Encoder for task-specific information."""
    def __init__(self, num_tasks, embedding_dim, hidden_dim, dropout=0.1):
        super().__init__()
        
        self.task_embedding = nn.Embedding(num_tasks + 1, embedding_dim)
        self.task_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, task_ids):
        task_ids = task_ids.squeeze(-1)
        task_emb = self.task_embedding(task_ids)
        return self.task_projection(task_emb)


class FeatureEncoder(nn.Module):
    """Encoder for static and dynamic features."""
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        
        self.feature_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Normalize and encode input features
        x = self.feature_norm(x)
        return self.encoder(x)


class HATNet(BaseModel):
    """Hierarchical Attention Network with Temporal Convolutions."""
    def __init__(
        self,
        input_size=32,
        hidden_size=128,
        tcn_channels=[64, 128],
        num_tasks=34,
        task_embedding_dim=32,
        attention_dim=64,
        dropout=0.3,
        dropout_tcn=0.2,
        verbose=False
    ):
        super().__init__()
        
        self.save_hyperparameters()
        self.verbose = verbose
        
        # Feature encoder
        self.feature_encoder = FeatureEncoder(
            input_dim=input_size,
            hidden_dim=hidden_size,
            dropout=dropout
        )
        
        # Task encoder
        self.task_encoder = TaskEncoder(
            num_tasks=num_tasks,
            embedding_dim=task_embedding_dim,
            hidden_dim=hidden_size,
            dropout=dropout
        )
        
        # Temporal Convolutional Network
        self.tcn = TemporalConvNet(
            in_channels=hidden_size,
            channels=tcn_channels,
            kernel_size=3,
            dropout=dropout_tcn
        )
        
        # Hierarchical attention
        self.hierarchical_attention = HierarchicalAttention(
            input_dim=tcn_channels[-1],
            attention_dim=attention_dim,
            dropout=dropout
        )
        
        # Fusion layer that combines task embedding and attended features
        self.fusion_layer = nn.Sequential(
            nn.Linear(tcn_channels[-1] + hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification layer
        self.classifier = nn.Linear(hidden_size, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability."""
        # Initialize task embedding with smaller values
        nn.init.uniform_(self.task_encoder.task_embedding.weight, -0.05, 0.05)
        
        # Initialize classifier with small weights for stability
        nn.init.xavier_uniform_(self.classifier.weight, gain=0.01)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x, task_ids, masks=None):
        """Forward pass of HATNet.
        
        Args:
            x: Input features (batch_size, seq_len, input_dim)
            task_ids: Task identifiers (batch_size, 1)
            masks: Optional masks for padding (batch_size, seq_len)
            
        Returns:
            Classification logits (batch_size, 1)
        """
        batch_size, seq_len, _ = x.size()
        
        # Handle missing values and outliers
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        x = torch.clamp(x, -10, 10)
        
        # Encode input features 
        x = self.feature_encoder(x)
        
        # Encode task information
        task_encoding = self.task_encoder(task_ids)
        
        # Apply temporal convolutions for feature extraction
        x_tcn = self.tcn(x)
        
        # Apply hierarchical attention
        task_context, attention_weights = self.hierarchical_attention(x_tcn, task_ids, masks)
        
        # Store attention weights for visualization during training
        if self.training:
            self.last_attention_weights = attention_weights
        
        # Combine task encoding with stroke features
        combined = torch.cat([task_context, task_encoding], dim=1)
        fused = self.fusion_layer(combined)
        
        # Final classification
        logits = self.classifier(fused)
        
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
        """Aggregate window-level predictions to subject-level with confidence weighting."""
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
            
            # Higher confidence weights for predictions farther from decision boundary
            confidence = np.abs(preds_array - 0.5) + 0.5
            weighted_avg = np.average(preds_array, weights=confidence)
            
            final_preds[subj] = weighted_avg > 0.5
            
        return final_preds