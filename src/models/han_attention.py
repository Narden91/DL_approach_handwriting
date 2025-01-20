import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureLevelAttention(nn.Module):
    """Attention mechanism for feature level processing."""
    def __init__(self, feature_dim, attention_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, feature_dim)
            mask: Optional mask tensor (batch_size, seq_len)
        """
        # Calculate attention scores
        scores = self.attention(x)  # (batch_size, seq_len, 1)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        weights = torch.softmax(scores, dim=1)
        
        # Apply attention weights to input
        context = torch.sum(weights * x, dim=1)
        return context, weights


class StrokeLevelAttention(nn.Module):
    """Attention mechanism for stroke level processing."""
    def __init__(self, hidden_dim, attention_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )
        
    def forward(self, x, stroke_mask=None):
        """
        Args:
            x: Input tensor (batch_size, num_strokes, hidden_dim)
            stroke_mask: Optional mask tensor (batch_size, num_strokes)
        """
        scores = self.attention(x)
        
        if stroke_mask is not None:
            scores = scores.masked_fill(stroke_mask.unsqueeze(-1) == 0, -1e9)
            
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context, weights


class TaskLevelAttention(nn.Module):
    """Attention mechanism for task level processing."""
    def __init__(self, task_dim, attention_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(task_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )
        
    def forward(self, x, task_mask=None):
        """
        Args:
            x: Input tensor (batch_size, num_tasks, task_dim)
            task_mask: Optional mask tensor (batch_size, num_tasks)
        """
        scores = self.attention(x)
        
        if task_mask is not None:
            scores = scores.masked_fill(task_mask.unsqueeze(-1) == 0, -1e9)
            
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context, weights


class HANEncoder(nn.Module):
    """Hierarchical Attention Network Encoder."""
    def __init__(
        self,
        static_dim,
        dynamic_dim,
        hidden_dim,
        attention_dim,
        num_tasks,
        dropout=0.1
    ):
        super().__init__()
        
        # Feature encoders
        self.static_encoder = nn.Linear(static_dim, hidden_dim)
        self.dynamic_encoder = nn.Linear(dynamic_dim, hidden_dim)
        
        # Attention modules
        self.feature_attention = FeatureLevelAttention(hidden_dim, attention_dim)
        self.stroke_attention = StrokeLevelAttention(hidden_dim, attention_dim)
        self.task_attention = TaskLevelAttention(hidden_dim, attention_dim)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, static_features, dynamic_features, task_ids, masks):
        """
        Args:
            static_features: Static features tensor (batch_size, static_dim)
            dynamic_features: Dynamic features tensor (batch_size, seq_len, dynamic_dim)
            task_ids: Task identifiers tensor (batch_size, 1)
            masks: Feature masks tensor (batch_size, seq_len)
        """
        batch_size = static_features.size(0)
        
        # Process static features
        static_encoded = self.static_encoder(static_features)
        static_encoded = self.layer_norm(static_encoded)
        static_encoded = self.dropout(static_encoded)
        
        # Process dynamic features
        dynamic_encoded = self.dynamic_encoder(dynamic_features)
        dynamic_encoded = self.layer_norm(dynamic_encoded)
        dynamic_encoded = self.dropout(dynamic_encoded)
        
        # Feature-level attention
        feature_context, feature_weights = self.feature_attention(
            dynamic_encoded, masks
        )
        
        # Combine static and dynamic features
        combined_features = torch.cat([static_encoded, feature_context], dim=-1)
        
        # Task-specific processing
        task_encoded = torch.zeros_like(combined_features).scatter_(
            1, task_ids, combined_features
        )
        
        # Task-level attention
        task_context, task_weights = self.task_attention(task_encoded)
        
        return task_context, {
            'feature_weights': feature_weights,
            'task_weights': task_weights
        }


class HANDecoder(nn.Module):
    """Decoder for Hierarchical Attention Network."""
    def __init__(self, context_dim, hidden_dim, dropout=0.1):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(context_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, context):
        """
        Args:
            context: Context vector from encoder
        """
        return self.decoder(context)