import torch
import torch.nn as nn
import numpy as np
from rich import print as rprint

from src.models.base import BaseModel
from src.models.han_attention import HANEncoder, HANDecoder, FeatureLevelAttention


class HANModel(BaseModel):
    def __init__(
        self,
        static_features,
        dynamic_features,
        hidden_size=128,
        attention_dim=64,
        num_tasks=34,
        dropout=0.3,
        verbose=False
    ):
        super().__init__()
        
        self.save_hyperparameters(ignore=['static_features', 'dynamic_features'])
        self.verbose = verbose
        
        # Store feature dimensions
        self.static_dim = len(static_features)
        self.dynamic_dim = len(dynamic_features)
        
        # Feature normalization
        self.static_norm = nn.LayerNorm(self.static_dim)
        self.dynamic_norm = nn.LayerNorm(self.dynamic_dim)
        
        # Feature encoders
        self.static_encoder = nn.Sequential(
            nn.Linear(self.static_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(self.dynamic_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task embedding
        self.task_embedding = nn.Embedding(num_tasks + 1, hidden_size)
        
        # Attention mechanisms
        self.feature_attention = FeatureLevelAttention(hidden_size, attention_dim)
        
        # Final classifier
        classifier_input_size = hidden_size * 2  # static + dynamic context
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        # Store feature names
        self.static_features = static_features
        self.dynamic_features = dynamic_features
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with appropriate initialization schemes."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param, gain=0.1)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(self, x, task_ids, masks=None):
        """
        Forward pass of the model.
        Args:
            x: Input tensor (batch_size, seq_len, total_features)
            task_ids: Task identifiers (batch_size, 1)
            masks: Optional mask tensor (batch_size, seq_len)
        """
        batch_size = x.size(0)
        
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
        
        if self.verbose:
            rprint(f"[blue]Static encoded shape: {static_encoded.shape}[/blue]")
            rprint(f"[blue]Dynamic encoded shape: {dynamic_encoded.shape}[/blue]")
        
        # Get dynamic context using attention
        dynamic_context, attention_weights = self.feature_attention(dynamic_encoded, masks)
        
        if self.verbose:
            rprint(f"[blue]Dynamic context shape: {dynamic_context.shape}[/blue]")
        
        # Combine static and dynamic features
        combined_features = torch.cat([static_encoded, dynamic_context], dim=1)
        
        if self.verbose:
            rprint(f"[blue]Combined features shape: {combined_features.shape}[/blue]")
        
        # Final classification
        output = self.classifier(combined_features)
        
        if self.verbose:
            rprint(f"[blue]Output shape: {output.shape}[/blue]")
            
        return output
    
    def _log_attention_weights(self, attention_weights):
        """Log attention weights for interpretability."""
        feature_weights = attention_weights['feature_weights']
        task_weights = attention_weights['task_weights']
        
        rprint("\n[bold blue]Attention Weights:[/bold blue]")
        
        # Log feature attention
        rprint("\nFeature Attention:")
        for i, weight in enumerate(feature_weights.mean(dim=0)):
            if i < len(self.dynamic_features):
                rprint(f"{self.dynamic_features[i]}: {weight.item():.4f}")
                
        # Log task attention
        rprint("\nTask Attention:")
        task_weights_mean = task_weights.mean(dim=0)
        for i, weight in enumerate(task_weights_mean):
            rprint(f"Task {i}: {weight.item():.4f}")
    
    def configure_optimizers(self):
        """Configure optimizer with warmup and cosine decay."""
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
        """Aggregate window-level predictions to subject-level with confidence."""
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