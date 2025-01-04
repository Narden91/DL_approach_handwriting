import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """Basic scaled dot-product attention mechanism."""
    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: Query (batch_size, n_heads, len_q, d_k)
            k: Key (batch_size, n_heads, len_k, d_k)
            v: Value (batch_size, n_heads, len_v, d_v)
            mask: Optional mask (batch_size, n_heads, len_q, len_k)
        """
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        
        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        
        residual = q
        
        # Pass through the pre-attention projection
        q = self.w_qs(q).view(batch_size, len_q, n_head, d_k).transpose(1, 2)
        k = self.w_ks(k).view(batch_size, len_k, n_head, d_k).transpose(1, 2)
        v = self.w_vs(v).view(batch_size, len_v, n_head, d_v).transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, n_head, -1, -1)
            
        output, attn = self.attention(q, k, v, mask=mask)
        
        # Concatenate and project back
        output = output.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        
        return output, attn


class TaskAwareAttention(nn.Module):
    """Task-aware attention mechanism that conditions on task embeddings."""
    def __init__(self, d_model, d_task, n_head=1, dropout=0.1):
        super().__init__()
        
        self.task_projection = nn.Linear(d_task, d_model)
        self.attention = MultiHeadAttention(
            n_head=n_head,
            d_model=d_model,
            d_k=d_model // n_head,
            d_v=d_model // n_head,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, x, task_embedding, mask=None):
        """
        Args:
            x: Input sequence (batch_size, seq_len, d_model)
            task_embedding: Task embedding (batch_size, d_task)
            mask: Optional mask (batch_size, seq_len)
        """
        batch_size = x.size(0)
        
        # Project task embedding to the same dimension as x
        task_query = self.task_projection(task_embedding).unsqueeze(1)
        
        # Use task embedding as query, sequence as key and value
        output, attn = self.attention(task_query, x, x, mask=mask)
        
        return output.squeeze(1), attn


class TemporalAttention(nn.Module):
    """Temporal attention module for capturing time-dependent patterns."""
    def __init__(self, d_model, window_size, n_head=1, dropout=0.1):
        super().__init__()
        
        self.position_enc = PositionalEncoding(d_model, max_len=window_size)
        self.self_attention = MultiHeadAttention(
            n_head=n_head,
            d_model=d_model,
            d_k=d_model // n_head,
            d_v=d_model // n_head,
            dropout=dropout
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input sequence (batch_size, seq_len, d_model)
            mask: Optional mask (batch_size, seq_len)
        """
        # Add positional encoding
        x = self.position_enc(x)
        
        # Self attention
        output, attn = self.self_attention(x, x, x, mask=mask)
        
        return output, attn


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal attention."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]