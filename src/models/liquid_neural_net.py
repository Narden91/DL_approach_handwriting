import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models.base import BaseModel


class LiquidCell(nn.Module):
    """
    Liquid Neural Network Cell with adaptive time constants
    """
    def __init__(self, input_size, hidden_size, tau_init=1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Neural ODE parameters
        self.W_ih = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Adaptive time constant parameters
        self.tau = nn.Parameter(torch.ones(hidden_size) * tau_init)
        self.tau_min = 0.1
        self.tau_max = 10.0
        
        # Activation function
        self.activation = nn.Tanh()
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using orthogonal initialization"""
        nn.init.orthogonal_(self.W_ih.weight, gain=0.1)
        nn.init.orthogonal_(self.W_hh.weight, gain=0.1)
        nn.init.zeros_(self.W_ih.bias)
        nn.init.zeros_(self.W_hh.bias)
        
    def forward(self, x, h, dt):
        """
        Forward pass implementing liquid time-constant dynamics
        
        Args:
            x: Input tensor (batch_size, input_size)
            h: Hidden state (batch_size, hidden_size)
            dt: Time step size
        """
        # Clamp tau values
        tau = torch.clamp(self.tau, self.tau_min, self.tau_max)
        
        # Compute dynamics
        dh = (self.W_ih(x) + self.W_hh(h))
        dh = self.activation(dh)
        
        # Update hidden state using liquid time-constant
        h_new = h + (dt / tau) * (dh - h)
        
        return h_new


class LiquidLayer(nn.Module):
    """Layer of Liquid Neural Network cells"""
    def __init__(self, input_size, hidden_size, num_cells=1):
        super().__init__()
        self.cells = nn.ModuleList([
            LiquidCell(input_size, hidden_size)
            for _ in range(num_cells)
        ])
        
    def forward(self, x, h, dt):
        """
        Forward pass through all liquid cells
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            h: Hidden states list [(batch_size, hidden_size)]
            dt: Time step size
        """
        new_h = []
        for i, cell in enumerate(self.cells):
            new_h.append(cell(x, h[i], dt))
        return new_h

class LiquidNeuralNetwork(BaseModel):
    """
    Liquid Neural Network implementation compatible with existing project structure
    """
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
        verbose=False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.verbose = verbose
        self.dt = dt
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(input_size)
        
        # Task embedding
        self.task_embedding = nn.Embedding(num_tasks + 1, task_embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Combined input size
        liquid_input_size = input_size + task_embedding_dim
        
        # Liquid layers
        self.layers = nn.ModuleList()
        self.layers.append(LiquidLayer(liquid_input_size, hidden_size, num_cells))
        
        for _ in range(num_layers - 1):
            self.layers.append(LiquidLayer(hidden_size, hidden_size, num_cells))
            
        # Output classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with appropriate initialization schemes"""
        # Initialize embedding
        nn.init.uniform_(self.task_embedding.weight, -0.05, 0.05)
        
        # Initialize classifier
        for name, param in self.classifier.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param, gain=0.1)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(self, x, task_ids, masks=None):
        """
        Forward pass of the liquid neural network
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            task_ids: Task identifiers (batch_size, 1)
            masks: Optional mask tensor (batch_size, seq_len)
        """
        batch_size, seq_len, _ = x.size()
        
        # Initial preprocessing
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        x = torch.clamp(x, -10, 10)
        x = self.feature_norm(x)
        
        # Process task embeddings
        task_ids = task_ids.squeeze(-1)
        task_emb = self.task_embedding(task_ids)
        task_emb = self.embedding_dropout(task_emb)
        task_emb = task_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine features with task embeddings
        x = torch.cat([x, task_emb], dim=-1)
        
        # Apply masking if provided
        if masks is not None:
            x = x * masks.unsqueeze(-1)
        
        # Process through liquid layers
        h_states = [[torch.zeros(batch_size, self.hparams.hidden_size, device=x.device)
                    for _ in range(layer.cells.__len__())]
                   for layer in self.layers]
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            for i, layer in enumerate(self.layers):
                h_states[i] = layer(x_t, h_states[i], self.dt)
                x_t = torch.stack(h_states[i]).mean(dim=0)
        
        # Use final hidden state for classification
        final_state = torch.stack(h_states[-1]).mean(dim=0)
        return self.classifier(final_state)
    
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
        
        # Average predictions for each subject
        final_preds = {subj: np.mean(preds) > 0.5 
                      for subj, preds in subject_preds.items()}
        return final_preds