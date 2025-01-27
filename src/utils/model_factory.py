from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import torch.nn as nn
from rich import print as rprint

@dataclass
class BaseModelConfig:
    """Base configuration class containing common parameters for all models."""
    input_size: int
    hidden_size: int = 128
    task_embedding_dim: int = 32
    num_tasks: int = 34
    dropout: float = 0.3
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {k: v for k, v in self.__dict__.items()}

class ModelFactory:
    """Factory class for creating neural network model instances.
    
    This class handles the creation of various neural network architectures for 
    handwriting analysis. It manages model-specific configurations and ensures
    proper parameter handling for each architecture type.
    """
    
    @staticmethod
    def _get_feature_split(feature_cols: List[str]) -> tuple:
        """Split features into static and dynamic categories.
        
        This method separates demographic (static) features from time-series
        (dynamic) features, which is particularly important for hierarchical
        attention networks and similar architectures.
        
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
    def create_model(cfg: Dict[str, Any], data_module: Any) -> nn.Module:
        """Create and return a model instance based on configuration.
        
        This method serves as the main factory method, creating the appropriate
        neural network model based on the provided configuration. It handles
        the specific parameter requirements of each model architecture.
        
        Args:
            cfg: Configuration dictionary containing model parameters
            data_module: Data module instance containing dataset information
            
        Returns:
            Initialized model instance
            
        Raises:
            ValueError: If unknown model type is specified
        """
        model_type = cfg.model.type.lower()
        
        if cfg.verbose:
            rprint(f"\n[bold blue]Creating model of type: {model_type}[/bold blue]")
        
        try:
            # Create base configuration dictionary with common parameters
            base_config = {
                'input_size': data_module.get_feature_dim(),
                'hidden_size': cfg.model.hidden_size,
                'task_embedding_dim': cfg.model.task_embedding_dim,
                'num_tasks': cfg.data.num_tasks,
                'dropout': cfg.model.dropout,
                'verbose': cfg.verbose
            }

            # Handle each model type with its specific requirements
            if model_type == "simplernn":
                # SimpleRNN uses a single-layer architecture with specialized regularization
                model_config = {
                    **base_config,
                    'embedding_dropout': cfg.model.embedding_dropout,
                    'zoneout_prob': cfg.model.zoneout_prob,
                    'activity_l1': cfg.model.activity_l1
                }
                from src.models.simpleRNN import SimpleRNN
                return SimpleRNN(**model_config)

            elif model_type == "rnn":
                # Standard RNN with configurable number of layers
                model_config = {
                    **base_config,
                    'num_layers': cfg.model.num_layers,
                    'nonlinearity': cfg.model.rnn_specific.nonlinearity
                }
                from src.models.RNN import RNN
                return RNN(**model_config)

            elif model_type == "lstm":
                # LSTM with layer normalization option
                model_config = {
                    **base_config,
                    'num_layers': cfg.model.num_layers,
                    'layer_norm': cfg.model.lstm_specific.layer_norm,
                    'bidirectional': cfg.model.bidirectional
                }
                from src.models.LSTM import LSTM
                return LSTM(**model_config)

            elif model_type == "gru":
                # GRU with bidirectional and batch processing options
                model_config = {
                    **base_config,
                    'num_layers': cfg.model.num_layers,
                    'batch_first': cfg.model.gru_specific.batch_first,
                    'bidirectional': cfg.model.bidirectional,
                    'bias': cfg.model.gru_specific.bias
                }
                from src.models.GRU import GRU
                return GRU(**model_config)

            elif model_type == "xlstm":
                # Extended LSTM with additional regularization options
                model_config = {
                    **base_config,
                    'num_layers': cfg.model.num_layers,
                    'layer_norm': cfg.model.lstm_specific.layer_norm,
                    'recurrent_dropout': cfg.model.xlstm_specific.recurrent_dropout,
                    'embedding_dropout': cfg.model.embedding_dropout
                }
                from src.models.XLSTM import XLSTM
                return XLSTM(**model_config)

            elif model_type == "attention_rnn":
                # Attention-enhanced RNN with multi-head attention
                # First, let's get the window size either from config or calculate from data module
                window_size = cfg.data.get('window_size', None)
                if window_size is None:
                    # If window_size isn't in config, use the one from data module's config
                    window_size = data_module.config.window_size if hasattr(data_module, 'config') else 25  # Default fallback
                    if cfg.verbose:
                        rprint(f"[yellow]Window size not found in config, using {window_size} from data module[/yellow]")
                
                model_config = {
                    **base_config,
                    'num_layers': cfg.model.num_layers,
                    'n_heads': cfg.model.attention_specific.n_heads,
                    'window_size': window_size  # Now we have a guaranteed value
                }
                from src.models.attention_RNN import AttentionRNN
                return AttentionRNN(**model_config)

            elif model_type == "han":
                # Get feature splits
                static_features, dynamic_features = ModelFactory._get_feature_split(
                    data_module.feature_cols
                )
                
                # Configure HANModel params
                model_config = {
                    'static_features': static_features,
                    'dynamic_features': dynamic_features,
                    'hidden_size': cfg.model.hidden_size,
                    'attention_dim': cfg.model.han_specific.attention_dim,
                    'num_tasks': cfg.data.num_tasks, 
                    'dropout': cfg.model.dropout,
                    'verbose': cfg.verbose
                }
                
                from src.models.han import HANModel
                return HANModel(**model_config)

            elif model_type == "lnn":
                model_config = {
                    **base_config,
                    'num_layers': cfg.model.num_layers,
                    'num_cells': cfg.model.lnn_specific.num_cells,
                    'dt': cfg.model.lnn_specific.dt,
                    'complexity_lambda': cfg.model.lnn_specific.get('complexity_lambda', 0.01)  # Default value 0.01
                }
                from src.models.liquid_neural_net import LiquidNeuralNetwork
                return LiquidNeuralNetwork(**model_config)

            else:
                raise ValueError(
                    f"Unknown model type: {model_type}. "
                    "Supported types are: rnn, lstm, gru, xlstm, simplernn, "
                    "attention_rnn, han, lnn"
                )

        except Exception as e:
            rprint(f"[red]Error creating model of type {model_type}: {str(e)}[/red]")
            raise