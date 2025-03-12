import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rich import print as rprint
from transformers import AutoModel, AutoConfig
from src.models.base import BaseModel


class HandwritingFeatureEmbedding(nn.Module):
    """Feature embedding layer for handwriting data.
    
    This module handles the embedding of feature vectors before they are processed
    by the transformer layers. It manages different types of features:
    - Continuous features: Normalized through layer normalization
    - Categorical features: Embedded through embedding layers
    - Temporal relationships: Encoded through positional embeddings
    """
    def __init__(
        self,
        input_size,
        embedding_dim,
        num_tasks,
        task_embedding_dim,
        dropout=0.3,
        max_seq_length=100
    ):
        super().__init__()
        
        # Feature normalization for continuous features
        self.feature_norm = nn.LayerNorm(input_size)
        
        # Task embedding
        self.task_embedding = nn.Embedding(num_tasks + 1, task_embedding_dim)
        
        # Positional encoding for temporal information
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        
        # Linear projection to transform features to embedding dimension
        self.feature_projection = nn.Linear(input_size, embedding_dim - task_embedding_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, task_ids, positions=None):
        """Forward pass for embedding features.
        
        Args:
            x: Feature tensor [batch_size, seq_len, input_size]
            task_ids: Task identifier tensor [batch_size, 1]
            positions: Optional position tensor [batch_size, seq_len]
            
        Returns:
            Embedded features [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Generate position indices if not provided
        if positions is None:
            positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
        
        # Normalize continuous features
        x = self.feature_norm(x)
        
        # Project features to embedding space
        x_projected = self.feature_projection(x)
        
        # Process task embeddings
        task_ids = task_ids.squeeze(-1)
        task_emb = self.task_embedding(task_ids).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Add positional embeddings
        pos_emb = self.position_embedding(positions)
        
        # Combine all embeddings
        combined_embedding = torch.cat([x_projected, task_emb], dim=-1) + pos_emb
        
        return self.dropout(combined_embedding)


class PretrainedTransformerModel(BaseModel):
    """Pretrained transformer model adapted for handwriting analysis.
    
    This model utilizes a pretrained transformer architecture and adapts it
    for the handwriting analysis task through fine-tuning and task-specific
    adaptations.
    """
    def __init__(
        self,
        input_size=32,
        hidden_size=256,
        num_heads=8,
        num_layers=6,
        num_tasks=34,
        task_embedding_dim=32,
        dropout=0.3,
        max_seq_length=100,
        pretrained_model_name="distilbert-base-uncased",
        freeze_base=False,
        verbose=False
    ):
        """Initialize pretrained transformer model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of the transformer hidden layers
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_tasks: Number of different tasks in the dataset
            task_embedding_dim: Dimension of task embeddings
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
            pretrained_model_name: Name of the pretrained model
            freeze_base: Whether to freeze the pretrained model
            verbose: Whether to print debug information
        """
        super().__init__()
        
        self.save_hyperparameters()
        self.verbose = verbose
        
        # Feature embedding layer
        self.feature_embedding = HandwritingFeatureEmbedding(
            input_size=input_size,
            embedding_dim=hidden_size,
            num_tasks=num_tasks,
            task_embedding_dim=task_embedding_dim,
            dropout=dropout,
            max_seq_length=max_seq_length
        )
        
        # Load or create transformer configuration
        try:
            # Try to load pretrained configuration
            if self.verbose:
                rprint(f"[blue]Loading pretrained model configuration: {pretrained_model_name}[/blue]")
            
            transformer_config = AutoConfig.from_pretrained(
                pretrained_model_name,
                hidden_size=hidden_size,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                intermediate_size=hidden_size*4,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout
            )
            
            # Load pretrained transformer model
            self.transformer = AutoModel.from_pretrained(
                pretrained_model_name,
                config=transformer_config
            )
            
            if self.verbose:
                rprint(f"[green]Successfully loaded pretrained model: {pretrained_model_name}[/green]")
                
        except Exception as e:
            # If loading fails, create a new transformer model with the configuration
            if self.verbose:
                rprint(f"[yellow]Could not load pretrained model: {str(e)}. Creating new transformer model.[/yellow]")
            
            transformer_config = AutoConfig.from_pretrained(
                pretrained_model_name,
                hidden_size=hidden_size,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                intermediate_size=hidden_size*4,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout
            )
            
            self.transformer = AutoModel.from_config(transformer_config)
        
        # Freeze pretrained model if specified
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
                
        # Reduce dimensionality gradually for more stable training
        self.pooling = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for linear layers and embeddings."""
        # Initialize feature embedding
        for module in [self.feature_embedding.feature_projection]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            nn.init.zeros_(module.bias)
        
        # Initialize pooling layer
        for idx, module in enumerate(self.pooling):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)
                
        # Initialize classifier with small weights
        for idx, module in enumerate(self.classifier):
            if isinstance(module, nn.Linear):
                gain = 0.01 if idx == len(self.classifier) - 1 else 0.1
                nn.init.xavier_uniform_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)
    
    def forward(self, x, task_ids, masks=None):
        """Forward pass of the transformer model.
        
        Args:
            x: Input feature tensor [batch_size, seq_len, input_size]
            task_ids: Task identifier tensor [batch_size, 1]
            masks: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Classification logits [batch_size, 1]
        """
        batch_size, seq_len, _ = x.size()
        
        # Handle missing values and outliers
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        x = torch.clamp(x, -10, 10)
        
        # Create positions and embed features
        positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
        embedded = self.feature_embedding(x, task_ids, positions)
        
        # Create attention mask from padding mask if provided
        attention_mask = masks
        
        # Pass through transformer
        transformer_outputs = self.transformer(
            inputs_embeds=embedded,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get sequence output and apply pooling
        sequence_output = transformer_outputs.last_hidden_state
        
        # Use attention mask for proper pooling if provided
        if attention_mask is not None:
            # Compute attention-weighted average
            mask_expanded = attention_mask.float().unsqueeze(-1)
            masked_output = sequence_output * mask_expanded
            # Sum and divide by the sum of the mask for masked average
            pooled_output = masked_output.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            # Use last token as the representative embedding if no mask
            pooled_output = sequence_output[:, -1]
        
        # Apply pooling and classification
        pooled_output = self.pooling(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def configure_optimizers(self):
        """Configure optimizer with layer-wise learning rate decay."""
        # Parameter groups with different learning rates
        no_decay = ['bias', 'LayerNorm.weight']
        
        # Group parameters: transformer has lower learning rate
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.transformer.named_parameters() 
                         if not any(nd in n for nd in no_decay)],
                'weight_decay': self.model_config['weight_decay'],
                'lr': self.model_config['learning_rate'] * 0.1  # Lower LR for pretrained model
            },
            {
                'params': [p for n, p in self.transformer.named_parameters() 
                         if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.model_config['learning_rate'] * 0.1  # Lower LR for pretrained model
            },
            {
                'params': [p for n, p in self.named_parameters() 
                         if 'transformer' not in n and not any(nd in n for nd in no_decay)],
                'weight_decay': self.model_config['weight_decay'],
                'lr': self.model_config['learning_rate']  # Full LR for other parts
            },
            {
                'params': [p for n, p in self.named_parameters() 
                         if 'transformer' not in n and any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.model_config['learning_rate']  # Full LR for other parts
            }
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.model_config['learning_rate'],
            eps=1e-8
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
            
            # Higher confidence weights for predictions farther from the decision boundary
            confidence = np.abs(preds_array - 0.5) + 0.5
            weighted_avg = np.average(preds_array, weights=confidence)
            
            final_preds[subj] = weighted_avg > 0.5
            
        return final_preds