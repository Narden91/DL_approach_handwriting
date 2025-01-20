from rich import print as rprint
from typing import Any, Dict, List

from src.models.RNN import RNN
from src.models.LSTM import LSTM
from src.models.XLSTM import XLSTM
from src.models.GRU import GRU
from src.models.attention_RNN import AttentionRNN
from src.models.simpleRNN import SimpleRNN
from src.models.han import HANModel
from src.models.liquid_neural_net import LiquidNeuralNetwork

class ModelFactory:
    """Factory class for creating model instances based on configuration."""
    
    @staticmethod
    def _get_feature_split(feature_cols: List[str]) -> tuple:
        """
        Split features into static and dynamic features.
        
        Args:
            feature_cols: List of all feature column names
            
        Returns:
            Tuple of (static_features, dynamic_features)
        """
        static_features = [
            col for col in feature_cols 
            if col in ['Age', 'Sex_encoded', 'Work_encoded']
        ]
        dynamic_features = [
            col for col in feature_cols 
            if col not in static_features
        ]
        return static_features, dynamic_features
    
    @staticmethod
    def create_model(cfg: Dict[str, Any], data_module: Any, window_size: Any) -> Any:
        """
        Create and return a model instance based on configuration.
        
        Args:
            cfg: Configuration dictionary containing model parameters
            data_module: Data module instance containing dataset information
            window_size: Size of the sliding window
        """
        model_type = cfg.model.type.lower()
        
        if cfg.verbose:
            rprint(f"\n[bold blue]Creating model of type: {model_type}[/bold blue]")
        
        try:
            if model_type == "lnn":
                return LiquidNeuralNetwork(
                    input_size=data_module.get_feature_dim(),
                    hidden_size=cfg.model.hidden_size,
                    num_layers=cfg.model.num_layers,
                    num_cells=cfg.model.lnn_specific.num_cells,
                    task_embedding_dim=cfg.model.task_embedding_dim,
                    num_tasks=cfg.data.num_tasks,
                    dropout=cfg.model.dropout,
                    dt=cfg.model.lnn_specific.dt,
                    verbose=cfg.verbose
                )
            
            elif model_type == "han":
                # Split features into static and dynamic
                static_features, dynamic_features = ModelFactory._get_feature_split(
                    data_module.feature_cols
                )
                
                if cfg.verbose:
                    rprint(f"\n[yellow]Static features ({len(static_features)}):[/yellow]")
                    for feat in static_features:
                        rprint(f"  - {feat}")
                    rprint(f"\n[yellow]Dynamic features ({len(dynamic_features)}):[/yellow]")
                    for feat in dynamic_features:
                        rprint(f"  - {feat}")
                
                return HANModel(
                    static_features=static_features,
                    dynamic_features=dynamic_features,
                    hidden_size=cfg.model.hidden_size,
                    attention_dim=cfg.model.han_specific.attention_dim,
                    num_tasks=cfg.data.num_tasks,
                    dropout=cfg.model.dropout,
                    verbose=cfg.verbose
                )
                
            elif model_type == "lstm":
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
                
            elif model_type == "attention_rnn":
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
            
    @staticmethod
    def get_model_specific_config(model_type: str) -> Dict[str, Any]:
        """
        Get model-specific configuration parameters.
        
        Args:
            model_type: Type of the model
            
        Returns:
            Dictionary with model-specific configuration
        """
        base_config = {
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'task_embedding_dim': 32,
        }
        
        model_specific = {
            'lstm': {
                'layer_norm': True,
            },
            'xlstm': {
                'layer_norm': True,
                'recurrent_dropout': 0.1,
            },
            'attention_rnn': {
                'n_heads': 4,
            },
            'gru': {
                'batch_first': True,
                'bidirectional': True,
                'bias': True,
            },
            'simplernn': {
                'zoneout_prob': 0.1,
                'activity_l1': 0.01,
            },
            'rnn': {
                'nonlinearity': 'tanh',
            },
            'han': {
                'attention_dim': 64,
            },
            'lnn': {
                'num_cells': 3,
                'dt': 0.1,
            }
        }
        
        model_type = model_type.lower()
        if model_type not in model_specific:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return {**base_config, **model_specific[model_type]}