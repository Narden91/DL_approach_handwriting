from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import torch.nn as nn
from rich import print as rprint
from src.models.GRU import GRU
from src.models.LSTM import LSTM
from src.models.XLSTM import XLSTM
from src.models.hat_net import HATNet
from src.models.simpleRNN import SimpleRNN
from src.models.RNN import RNN
from src.models.attention_RNN import AttentionRNN
from src.models.han import HandwritingHAN
from src.models.liquid_neural_net import LiquidNetwork
from src.models.transformer_model import PretrainedTransformerModel



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
                
                return SimpleRNN(**model_config)

            elif model_type == "rnn":
                # Standard RNN with configurable number of layers
                model_config = {
                    **base_config,
                    'num_layers': cfg.model.num_layers,
                    'nonlinearity': cfg.model.rnn_specific.nonlinearity
                }
                return RNN(**model_config)

            elif model_type == "lstm":
                # LSTM with layer normalization option
                model_config = {
                    **base_config,
                    'num_layers': cfg.model.num_layers,
                    'layer_norm': cfg.model.lstm_specific.layer_norm,
                    'use_attention': cfg.model.lstm_specific.use_attention,
                    'bidirectional': cfg.model.bidirectional
                }
                return LSTM(**model_config)

            elif model_type == "gru":
                # GRU with bidirectional and batch processing options
                model_config = {
                    **base_config,
                    'num_layers': cfg.model.num_layers,
                    'batch_first': cfg.model.gru_specific.batch_first,
                    'bidirectional': cfg.model.bidirectional,
                    'bias': cfg.model.gru_specific.bias,
                }
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
                    # 'window_size': window_size  # Now we have a guaranteed value
                }
                return AttentionRNN(**model_config)

            elif model_type == "han":
                # Get feature splits
                static_features, dynamic_features = ModelFactory._get_feature_split(
                    data_module.feature_cols
                )
                
                # Create feature dimensions dictionary
                feature_dims = {
                    'static': len(static_features),
                    'dynamic': len(dynamic_features)
                }
                
                # Configure HandwritingHAN params
                model_config = {
                    'feature_dims': feature_dims,
                    'hidden_size': cfg.model.hidden_size,
                    'attention_dim': cfg.model.han_specific.attention_dim,
                    'num_tasks': cfg.data.num_tasks,
                    'task_embedding_dim': cfg.model.task_embedding_dim,
                    'dropout': cfg.model.dropout,
                    'feature_dropout': cfg.model.han_specific.get('feature_dropout', 0.2),
                    'attention_dropout': cfg.model.han_specific.get('attention_dropout', 0.1),
                    'use_layer_norm': cfg.model.han_specific.get('use_layer_norm', True),
                    'verbose': cfg.verbose
                }
                return HandwritingHAN(**model_config)
            
            elif model_type == "hatnet":
                model_config = {
                    'input_size': data_module.get_feature_dim(),
                    'hidden_size': cfg.model.hidden_size,
                    'tcn_channels': [64, cfg.model.hidden_size],
                    'num_tasks': cfg.data.num_tasks,
                    'task_embedding_dim': cfg.model.task_embedding_dim,
                    'attention_dim': cfg.model.get('attention_dim', 64),
                    'dropout': cfg.model.dropout,
                    'dropout_tcn': cfg.model.get('dropout_tcn', 0.2),
                    'verbose': cfg.verbose
                }
                return HATNet(**model_config)

            elif model_type == "lnn":
                # Configure optimized LNN parameters
                model_config = {
                    'input_size': data_module.get_feature_dim(),
                    'hidden_size': cfg.model.hidden_size,
                    'num_layers': cfg.model.num_layers,
                    'num_tasks': cfg.data.num_tasks,
                    'task_embedding_dim': cfg.model.task_embedding_dim,
                    'dropout': cfg.model.dropout,
                    'dt': cfg.model.lnn_specific.get('dt', 0.1),  # Default dt value of 0.1
                    'bidirectional': cfg.model.get('bidirectional', True),  # Optional bidirectional processing
                    'verbose': cfg.verbose
                }
                return LiquidNetwork(**model_config)
            
            elif model_type == "transformer":
                # Configure optimized Transformer parameters
                model_config = {
                    'input_size': data_module.get_feature_dim(),
                    'hidden_size': cfg.model.hidden_size,
                    'num_heads': cfg.model.transformer_specific.get('num_heads', 8),
                    'num_layers': cfg.model.transformer_specific.get('num_layers', 6),
                    'num_tasks': cfg.data.num_tasks,
                    'task_embedding_dim': cfg.model.task_embedding_dim,
                    'dropout': cfg.model.dropout,
                    'max_seq_length': cfg.model.transformer_specific.get('max_seq_length', 100),
                    'pretrained_model_name': cfg.model.transformer_specific.get('pretrained_model_name', 'distilbert-base-uncased'),
                    'freeze_base': cfg.model.transformer_specific.get('freeze_base', False),
                    'verbose': cfg.verbose
                }
                return PretrainedTransformerModel(**model_config)

            else:
                raise ValueError(
                    f"Unknown model type: {model_type}. "
                    "Supported types are: rnn, lstm, gru, xlstm, simplernn, "
                    "attention_rnn, han, lnn, transformer"
                )

        except Exception as e:
            rprint(f"[red]Error creating model of type {model_type}: {str(e)}[/red]")
            raise