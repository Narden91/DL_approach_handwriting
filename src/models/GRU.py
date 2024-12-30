import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchmetrics import Accuracy, MatthewsCorrCoef, Precision, Recall, F1Score
from rich import print as rprint

from src.models.base import BaseModel
from src.models.RNN import RNNDebugger


class GRU(BaseModel):
    def __init__(
        self, 
        input_size=13,
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
        batch_first=True,
        bidirectional=True,
        bias=True,
        verbose=False
    ):
        super().__init__()
        
        # Input normalization layer
        self.input_norm = nn.BatchNorm1d(input_size, eps=1e-5, momentum=0.1)
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            bias=bias
        )
        
        # Final classifier
        self.classifier = nn.Linear(hidden_size * 2, 1)
        
        # Initialize weights
        self._init_weights()
        
        self.verbose = verbose
        self.debugger = RNNDebugger()
    
    def _init_weights(self):
        """
        Initialize GRU weights using Xavier/Glorot initialization
        with specific gains for each gate
        """
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                # Input weights
                nn.init.xavier_uniform_(param, gain=1.0)
            elif 'weight_hh' in name:
                # Hidden weights - slightly lower gain for stability
                nn.init.xavier_uniform_(param, gain=0.8)
            elif 'bias' in name:
                # Initialize biases to small positive values for reset/update gates
                # and zero for new gate
                param.data.fill_(0.0)
                param.data[self.gru.hidden_size:2*self.gru.hidden_size] = 1.0  # Update gate bias
    
    def forward(self, x, task_ids, masks):
        """
        Forward pass with enhanced stability checks
        """
        # Handle NaN and extreme values
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        x = torch.clamp(x, -10, 10)
        
        # Normalize input
        if self.verbose:
            self.debugger.check_tensor(x, "Pre-norm input", "Forward")
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        if self.verbose:
            self.debugger.check_tensor(x, "Post-norm input", "Forward")
        
        # Apply masking if provided
        if masks is not None:
            x = x * masks.unsqueeze(-1)
        
        # GRU forward pass
        outputs, _ = self.gru(x)
        if self.verbose:
            self.debugger.check_tensor(outputs, "GRU outputs", "Forward")
        
        # Apply dropout to final hidden states
        outputs = F.dropout(outputs, p=0.1, training=self.training)
        
        # Get final prediction
        final_output = self.classifier(outputs[:, -1, :])
        if self.verbose:
            self.debugger.check_tensor(final_output, "Final output", "Forward")
        
        return final_output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.model_config['learning_rate'],
            weight_decay=self.model_config['weight_decay']
        )
        
        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=self.verbose
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }