import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models.base import BaseModel
from src.models.attention_module import TaskAwareAttention, TaskSpecificTransformer, TemporalAttention


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        # q, k, v shapes: (batch_size, n_heads, len_q/k/v, d_k/v)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        
        return output, attn

class MultiHeadAttention(nn.Module):
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

class AttentionRNN(BaseModel):
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
        window_size=25,
        verbose=False
    ):
        super().__init__()
        self.verbose = verbose

        # Feature normalization
        self.feature_norm = nn.LayerNorm(input_size)

        # Task embedding
        self.task_embedding = nn.Embedding(num_tasks + 1, task_embedding_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        # Projection layer to ensure task embedding matches hidden_size (128)
        self.task_projection = nn.Linear(task_embedding_dim, hidden_size)

        # Input projection to match hidden size before passing to transformer
        self.input_projection = nn.Linear(input_size, hidden_size)

        # Initialize Task-Specific Transformer
        self.task_transformer = TaskSpecificTransformer(
            d_model=hidden_size,
            d_task=hidden_size,  # Ensure task embedding is projected to hidden size
            num_heads=n_heads,
            dropout=dropout
        )

        # RNN layer
        self.rnn = nn.RNN(
            input_size=hidden_size,  # Already projected
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0 if num_layers == 1 else dropout
        )

        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Output classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        self._init_weights()
        
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
    def _init_weights(self):
        """Initialize weights with appropriate initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param, gain=0.1)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
        # Special initialization for embedding
        nn.init.uniform_(self.task_embedding.weight, -0.05, 0.05)

    def forward(self, x, task_ids, masks=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            task_ids: Task identifiers (batch_size, 1)
            masks: Optional mask tensor (batch_size, seq_len)
        
        Returns:
            Output logits for classification
        """
        device = x.device  # Ensure everything is on the same device

        # Move tensors to device
        task_ids = task_ids.squeeze(-1).to(device)  
        task_emb = self.task_embedding(task_ids).to(device)  
        task_emb = self.embedding_dropout(task_emb)

        # Ensure task embedding matches hidden_size (d_model)
        task_emb = self.task_projection(task_emb)  

        # Project input x to hidden size before passing to the transformer
        x = self.input_projection(x)

        # Apply Task-Specific Transformer
        x = self.task_transformer(x, task_emb)

        # RNN processing
        rnn_out, _ = self.rnn(x)

        # Self-attention
        attention_mask = ~masks.bool() if masks is not None else None
        attended_output, _ = self.self_attention(
            rnn_out, rnn_out, rnn_out,
            key_padding_mask=attention_mask
        )

        # Use the final output for classification
        final_output = attended_output[:, -1, :].to(device)

        # print(f"Device Check -> x: {x.device}, task_emb: {task_emb.device}, rnn_out: {rnn_out.device}")
        # print(f"Attention Output: {attended_output.device}, Final Output: {final_output.device}")

        return self.classifier(final_output).to(device)

    def configure_optimizers(self):
        """Configure optimizer with warmup and cosine decay"""
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
        """Aggregate window-level predictions to subject-level with confidence"""
        pred_probs = torch.sigmoid(window_preds).cpu().numpy()
        subject_preds = {}
        
        for pred, subj_id in zip(pred_probs, subject_ids):
            if subj_id not in subject_preds:
                subject_preds[subj_id] = []
            subject_preds[subj_id].append(pred)
        
        # Weighted average based on prediction confidence
        final_preds = {}
        for subj, preds in subject_preds.items():
            preds = np.array(preds)
            confidence = np.abs(preds - 0.5) + 0.5  # Higher weight for more confident predictions
            weighted_avg = np.average(preds, weights=confidence)
            final_preds[subj] = weighted_avg > 0.5
            
        return final_preds