import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models.base import BaseModel


class TaskAwareAttention(nn.Module):
    """Attention mechanism that incorporates task-specific information.
    
    This module allows the model to focus on different aspects of the
    handwriting sequence based on the current task being performed.
    """
    def __init__(self, hidden_size, task_dim, num_heads=4, dropout=0.1):
        """Initialize task-aware attention mechanism.
        
        Args:
            hidden_size: Size of the hidden representations
            task_dim: Dimension of the task embedding
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        # Task-specific projection
        self.task_projection = nn.Linear(task_dim, hidden_size)
        
        # Multi-head attention components
        self.q_projection = nn.Linear(hidden_size, hidden_size)
        self.k_projection = nn.Linear(hidden_size, hidden_size)
        self.v_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states, task_embedding, mask=None):
        """Apply task-aware attention to the hidden states.
        
        Args:
            hidden_states: Sequence of hidden states (batch_size, seq_len, hidden_size)
            task_embedding: Task embedding vector (batch_size, task_dim)
            mask: Optional attention mask (batch_size, seq_len)
            
        Returns:
            Attended hidden states (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        # Project task embedding
        task_info = self.task_projection(task_embedding).unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        # Use task embedding to modulate query projection
        q = self.q_projection(hidden_states + task_info)
        k = self.k_projection(hidden_states)
        v = self.v_projection(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, v)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # Final projection and residual connection
        output = self.output_projection(context)
        output = self.dropout(output)
        output = self.layer_norm(output + hidden_states)
        
        return output


class AttentionRNN(BaseModel):
    """RNN model with task-aware attention for handwriting analysis.
    
    This model combines a recurrent neural network with a task-aware
    attention mechanism to better capture relevant patterns in handwriting
    sequences for different tasks.
    """
    def __init__(
        self,
        input_size=13,
        hidden_size=128,
        num_layers=2,
        task_embedding_dim=32,
        num_tasks=34,
        n_heads=4,
        dropout=0.3,
        embedding_dropout=0.1,
        bidirectional=True,
        verbose=False
    ):
        """Initialize the AttentionRNN model.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Size of RNN hidden states
            num_layers: Number of RNN layers
            task_embedding_dim: Dimension of task embeddings
            num_tasks: Number of different tasks
            n_heads: Number of attention heads
            dropout: Dropout probability
            embedding_dropout: Dropout for embeddings
            bidirectional: Whether to use bidirectional RNN
            verbose: Whether to print debug information
        """
        super().__init__()
        self.verbose = verbose
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Feature normalization
        self.feature_norm = nn.LayerNorm(input_size)

        # Task embedding
        self.task_embedding = nn.Embedding(num_tasks + 1, task_embedding_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        # RNN layer
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output size considering bidirectionality
        output_size = hidden_size * 2 if bidirectional else hidden_size

        # Task-aware attention
        self.task_attention = TaskAwareAttention(
            hidden_size=output_size,
            task_dim=task_embedding_dim,
            num_heads=n_heads,
            dropout=dropout
        )

        # Output classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Dropout(dropout),
            nn.Linear(output_size, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with appropriate initialization methods."""
        # Initialize task embedding
        nn.init.uniform_(self.task_embedding.weight, -0.05, 0.05)
        
        # Initialize RNN weights
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param, gain=1.0)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize classifier
        for name, param in self.classifier.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x, task_ids, masks=None):
        """
        Forward pass of the AttentionRNN model.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            task_ids: Task identifiers (batch_size, 1)
            masks: Optional mask tensor (batch_size, seq_len)
        
        Returns:
            Output logits for classification
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
        task_emb = self.embedding_dropout(task_emb)

        # Initialize hidden state
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )

        # RNN processing
        rnn_out, _ = self.rnn(x, h0)

        # Apply task-aware attention
        attended_output = self.task_attention(rnn_out, task_emb, masks)

        # Use the final output for classification
        final_output = attended_output[:, -1, :]

        return self.classifier(final_output)

    def configure_optimizers(self):
        """Configure optimizer with cosine annealing scheduler."""
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
                "monitor": "val_f1"
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
        
        # Weight predictions by confidence
        final_preds = {}
        for subj, preds in subject_preds.items():
            preds = np.array(preds)
            confidence = np.abs(preds - 0.5) + 0.5
            weighted_avg = np.average(preds, weights=confidence)
            final_preds[subj] = weighted_avg > 0.5
            
        return final_preds