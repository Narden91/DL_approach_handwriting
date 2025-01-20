import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models.base import BaseModel

class AdaptiveLiquidCell(nn.Module):
    def __init__(self, input_size, hidden_size, task_embedding_dim, tau_init=1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Neural ODE parameters - adjusted input size
        self.W_ih = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Adaptive time constant parameters
        self.tau = nn.Parameter(torch.ones(hidden_size) * tau_init)
        self.tau_min = 0.1
        self.tau_max = 10.0
        
        # Task modulation with correct dimensions
        self.task_modulation = nn.Linear(task_embedding_dim, hidden_size)
        self.task_gate = nn.Sequential(
            nn.Linear(task_embedding_dim, hidden_size),
            nn.Sigmoid()
        )
        
        # dt predictor with correct dimensions
        self.dt_predictor = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def _init_weights(self):
        """Initialize weights using orthogonal initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param, gain=0.1)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(self, x, h, task_emb, base_dt):
        """
        Forward pass with adaptive dynamics and task modulation
        
        Args:
            x: Input tensor (batch_size, input_size)
            h: Hidden state (batch_size, hidden_size)
            task_emb: Task embedding (batch_size, task_embedding_dim)
            base_dt: Base time step size
        """
        # Predict adaptive dt
        combined = torch.cat([x, h], dim=-1)
        dt_scale = self.dt_predictor(combined)
        dt = base_dt * dt_scale
        
        # Task modulation
        task_mod = self.task_modulation(task_emb)
        task_gate = self.task_gate(task_emb)
        
        # Compute dynamics with task modulation
        dh = (self.W_ih(x) + self.W_hh(h)) * task_gate + task_mod
        
        # Apply layer normalization
        dh = self.layer_norm(dh)
        
        # Clamp tau values
        tau = torch.clamp(self.tau, self.tau_min, self.tau_max)
        
        # Update hidden state using liquid time-constant
        h_new = h + (dt / tau) * (dh - h)
        
        return h_new, dt_scale

class ResidualLiquidLayer(nn.Module):
    def __init__(self, input_size, hidden_size, task_embedding_dim, num_cells=3):
        super().__init__()
        self.cells = nn.ModuleList([
            AdaptiveLiquidCell(input_size, hidden_size, task_embedding_dim)
            for _ in range(num_cells)
        ])
        
        self.skip_connection = nn.Linear(input_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.hidden_size = hidden_size
        
        # Hidden state buffer
        self.h_buffer = None
        
    def init_hidden(self, batch_size, device):
        """Initialize hidden state buffer"""
        self.h_buffer = [
            torch.zeros(batch_size, self.hidden_size, device=device)
            for _ in range(len(self.cells))
        ]
        
    def monitor_stability(self, h_states, threshold=10.0):
        """Monitor hidden states for numerical stability"""
        max_vals = torch.stack([torch.max(abs(h)) for h in h_states])
        return torch.any(max_vals > threshold)
        
    def forward(self, x, task_emb, dt):
        """
        Forward pass through all liquid cells with residual connection
        
        Args:
            x: Input tensor (batch_size, input_size)
            task_emb: Task embedding (batch_size, task_embedding_dim)
            dt: Base time step size
        """
        if self.h_buffer is None:
            self.init_hidden(x.size(0), x.device)
            
        new_h = []
        dt_scales = []
        
        for i, cell in enumerate(self.cells):
            h_new, dt_scale = cell(x, self.h_buffer[i], task_emb, dt)
            new_h.append(h_new)
            dt_scales.append(dt_scale)
            
        # Update buffer
        self.h_buffer = new_h
        
        # Compute layer output with residual connection
        liquid_out = torch.stack(new_h).mean(dim=0)
        skip_out = self.skip_connection(x)
        output = self.layer_norm(liquid_out + skip_out)
        
        return output, torch.stack(dt_scales).mean()

class LiquidNeuralNetwork(BaseModel):
    def __init__(
        self,
        input_size=13,
        hidden_size=128,
        num_layers=2,
        num_cells=3,
        task_embedding_dim=32,
        num_tasks=34,
        dropout=0.3,
        dt=0.1,
        complexity_lambda=0.01,
        verbose=False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.verbose = verbose
        self.dt = dt
        self.complexity_lambda = complexity_lambda
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(input_size)
        
        # Task embedding
        self.task_embedding = nn.Embedding(num_tasks + 1, task_embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Important: Don't combine input size with task embedding dim here
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Liquid layers with corrected dimensions
        self.layers = nn.ModuleList([
            ResidualLiquidLayer(
                hidden_size,  # Input size is now hidden_size for all layers
                hidden_size,
                task_embedding_dim,
                num_cells
            )
            for i in range(num_layers)
        ])
        
        # Output classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with appropriate initialization schemes"""
        nn.init.uniform_(self.task_embedding.weight, -0.05, 0.05)
        
        for name, param in self.classifier.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param, gain=0.1)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def compute_complexity_loss(self):
        """Compute complexity regularization loss"""
        complexity = 0
        for layer in self.layers:
            for cell in layer.cells:
                complexity += torch.norm(cell.tau)
        return complexity * self.complexity_lambda
        
    def forward(self, x, task_ids, masks=None):
        batch_size, seq_len, _ = x.size()
        
        # Initial preprocessing
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        x = torch.clamp(x, -10, 10)
        x = self.feature_norm(x)
        
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Process task embeddings
        task_ids = task_ids.squeeze(-1)
        task_emb = self.task_embedding(task_ids)
        task_emb = self.embedding_dropout(task_emb)
        
        # Initialize stability monitoring
        is_unstable = False
        current_dt = self.dt
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Reset layer buffers at sequence start
            if t == 0:
                for layer in self.layers:
                    layer.h_buffer = None
            
            # Process through liquid layers
            layer_input = x_t
            dt_scales = []
            
            for layer in self.layers:
                layer_output, dt_scale = layer(layer_input, task_emb, current_dt)
                dt_scales.append(dt_scale)
                
                if layer.monitor_stability(layer.h_buffer):
                    is_unstable = True
                    current_dt *= 0.5
                
                layer_input = layer_output
            
            outputs.append(layer_input)
            
            if masks is not None:
                outputs[-1] = outputs[-1] * masks[:, t].unsqueeze(-1)
        
        # Combine outputs with attention
        outputs = torch.stack(outputs, dim=1)
        if masks is not None:
            attention_weights = F.softmax(masks.float(), dim=1)
            final_output = (outputs * attention_weights.unsqueeze(-1)).sum(dim=1)
        else:
            final_output = outputs[:, -1]
        
        # Classification
        logits = self.classifier(final_output)
        
        if self.training:
            self.complexity_loss = self.compute_complexity_loss()
            
        return logits
    
    def training_step(self, batch, batch_idx):
        """Override training step to include complexity regularization"""
        loss = super().training_step(batch, batch_idx)
        if hasattr(self, 'complexity_loss'):
            loss = loss + self.complexity_loss
        return loss
    
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
                "monitor": "val_f1",
                "frequency": 1
            }
        }
    
    def aggregate_predictions(self, window_preds, subject_ids):
        """Aggregate window-level predictions to subject-level"""
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
            confidence = np.abs(preds - 0.5) + 0.5
            weighted_avg = np.average(preds, weights=confidence)
            final_preds[subj] = weighted_avg > 0.5
            
        return final_preds