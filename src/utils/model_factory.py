from rich import print as rprint
from typing import Any, Dict

from src.models.RNN import RNN
from src.models.LSTM import LSTM
from src.models.XLSTM import XLSTM
from src.models.GRU import GRU
from src.models.attention_RNN import AttentionRNN
from src.models.simpleRNN import SimpleRNN


class ModelFactory:
    """Factory class for creating model instances based on configuration."""
    
    @staticmethod
    def create_model(cfg: Dict[str, Any], data_module: Any, window_size: Any) -> Any:
        """
        Create and return a model instance based on configuration.
        
        Args:
            cfg: Configuration dictionary containing model parameters
            data_module: Data module instance containing dataset information
            
        Returns:
            Instantiated model of specified type
            
        Raises:
            ValueError: If model type is not recognized
        """
        model_type = cfg.model.type.lower()
        
        # Log model creation
        if cfg.verbose:
            rprint(f"\n[bold blue]Creating model of type: {model_type}[/bold blue]")
        
        try:
            if model_type == "lstm":
                return LSTM(
                    input_size=data_module.get_feature_dim(),
                    hidden_size=cfg.model.hidden_size,
                    num_layers=cfg.model.num_layers,
                    dropout=cfg.model.dropout,
                    layer_norm=cfg.model.lstm_specific.layer_norm,
                    verbose=cfg.verbose
                )
                
            elif model_type == "xlstm":
                return XLSTM(
                    input_size=data_module.get_feature_dim(),
                    hidden_size=cfg.model.hidden_size,
                    num_layers=cfg.model.num_layers,
                    num_tasks=cfg.data.num_tasks,
                    task_embedding_dim=cfg.model.task_embedding_dim,
                    dropout=cfg.model.dropout,
                    embedding_dropout=cfg.model.embedding_dropout,
                    layer_norm=cfg.model.lstm_specific.layer_norm,
                    recurrent_dropout=cfg.model.xlstm_specific.recurrent_dropout,
                    verbose=cfg.verbose
                )
                
            elif cfg.model.type.lower() == "attention_rnn":
                return AttentionRNN(
                    input_size=data_module.get_feature_dim(),
                    hidden_size=cfg.model.hidden_size,
                    num_layers=cfg.model.num_layers, 
                    task_embedding_dim=cfg.model.task_embedding_dim,
                    num_tasks=cfg.data.num_tasks,
                    n_heads=cfg.model.attention_specific.n_heads,
                    dropout=cfg.model.dropout,
                    window_size=window_size,
                    verbose=cfg.verbose
                )
                
            elif model_type == "gru":
                return GRU(
                    input_size=data_module.get_feature_dim(),
                    hidden_size=cfg.model.hidden_size,
                    num_layers=cfg.model.num_layers,
                    num_tasks=cfg.data.num_tasks,
                    task_embedding_dim=cfg.model.task_embedding_dim,
                    dropout=cfg.model.dropout,
                    batch_first=cfg.model.gru_specific.batch_first,
                    bidirectional=cfg.model.bidirectional,
                    bias=cfg.model.gru_specific.bias,
                    verbose=cfg.verbose
                )
                
            elif model_type == "simplernn":
                return SimpleRNN(
                    input_size=data_module.get_feature_dim(),
                    hidden_size=cfg.model.hidden_size,
                    task_embedding_dim=cfg.model.task_embedding_dim,
                    num_tasks=cfg.data.num_tasks,
                    dropout=cfg.model.dropout,
                    embedding_dropout=cfg.model.embedding_dropout,
                    zoneout_prob=cfg.model.zoneout_prob,
                    activity_l1=cfg.model.activity_l1,
                    verbose=cfg.verbose
                )
                
            elif model_type == "rnn":
                return RNN(
                    input_size=data_module.get_feature_dim(),
                    hidden_size=cfg.model.hidden_size,
                    num_layers=cfg.model.num_layers,
                    num_tasks=cfg.data.num_tasks,
                    task_embedding_dim=cfg.model.task_embedding_dim,
                    nonlinearity=cfg.model.rnn_specific.nonlinearity,
                    dropout=cfg.model.dropout,
                    verbose=cfg.verbose
                )
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            rprint(f"[red]Error creating model of type {model_type}: {str(e)}[/red]")
            raise