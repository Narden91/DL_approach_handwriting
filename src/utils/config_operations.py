import os
from typing import Any, Dict
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint
from rich.panel import Panel


class ConfigOperations:
    @staticmethod
    def get_env_config() -> Dict[str, Any]:
        """Retrieve configuration from environment variables."""
        def convert_value(key: str, value: str) -> Any:
            if value is None:
                return None
            if key in ["SEED", "NUM_FOLDS", "BATCH_SIZE", "MAX_EPOCHS", "EARLY_STOPPING_PATIENCE"]:
                return int(value)
            elif key in ["LEARNING_RATE", "WEIGHT_DECAY", "GRADIENT_CLIP_VAL"]:
                return float(value)
            elif key in ["VERBOSE", "TEST_MODE", "ENABLE_AUGMENTATION"]:  # ✅ Added enable_augmentation
                return value.lower() == "true"
            elif key in ["WINDOW_SIZES", "STRIDES"]:
                return [int(x) for x in value.split(',')]
            else:
                return value

        env_vars = {
            "seed": convert_value("SEED", os.getenv("SEED")),
            "verbose": convert_value("VERBOSE", os.getenv("VERBOSE")),
            "num_folds": convert_value("NUM_FOLDS", os.getenv("NUM_FOLDS")),
            "test_mode": convert_value("TEST_MODE", os.getenv("TEST_MODE")),
            "experiment_name": os.getenv("EXPERIMENT_NAME"),
            "data.window_sizes": convert_value("WINDOW_SIZES", os.getenv("WINDOW_SIZES")),
            "data.strides": convert_value("STRIDES", os.getenv("STRIDES")),
            "data.batch_size": convert_value("BATCH_SIZE", os.getenv("BATCH_SIZE")),
            "data.yaml_split_path": os.getenv("YAML_SPLIT_PATH"),
            "data.enable_augmentation": convert_value("ENABLE_AUGMENTATION", os.getenv("ENABLE_AUGMENTATION")),  # ✅ Added this
            "model.type": os.getenv("MODEL_TYPE"),
            "training.max_epochs": convert_value("MAX_EPOCHS", os.getenv("MAX_EPOCHS")),
            "training.learning_rate": convert_value("LEARNING_RATE", os.getenv("LEARNING_RATE")),
            "training.weight_decay": convert_value("WEIGHT_DECAY", os.getenv("WEIGHT_DECAY")),
            "training.early_stopping_patience": convert_value("EARLY_STOPPING_PATIENCE", os.getenv("EARLY_STOPPING_PATIENCE")),
            "training.gradient_clip_val": convert_value("GRADIENT_CLIP_VAL", os.getenv("GRADIENT_CLIP_VAL"))
        }
        # Remove None values
        return {k: v for k, v in env_vars.items() if v is not None}
    

    @staticmethod
    def merge_configurations(hydra_cfg: DictConfig) -> DictConfig:
        """Merge configurations from environment variables with Hydra config."""
        # Get environment variables
        env_config = ConfigOperations.get_env_config()

        # Create mutable configuration
        config = OmegaConf.create(OmegaConf.to_container(hydra_cfg, resolve=True))

        # Update with environment variables
        for key_path, value in env_config.items():
            parts = key_path.split('.')
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value

        # Display final configuration
        ConfigOperations.display_configuration(config, env_config.keys())

        return config

    @staticmethod
    def display_configuration(config: DictConfig, env_overrides: set) -> None:
        """Display the final configuration with highlighting for overridden values."""
        def format_value(path: str, value: Any) -> str:
            source = "env" if path in env_overrides else "default"
            return f"{value} [dim]({source})[/]"

        rprint(Panel(
            f"[bold green]Training Configuration[/]\n\n"
            f"[yellow]General Settings:[/]\n"
            f"  Experiment: {format_value('experiment_name', config.experiment_name)}\n"
            f"  Seed: {format_value('seed', config.seed)}\n"
            f"  Verbose: {format_value('verbose', config.verbose)}\n"
            f"  Num Folds: {format_value('num_folds', config.num_folds)}\n"
            f"  Test Mode: {format_value('test_mode', config.test_mode)}\n\n"
            f"[yellow]Data Settings:[/]\n"
            f"  Window sizes: {format_value('data.window_sizes', config.data.window_sizes)}\n"
            f"  Strides: {format_value('data.strides', config.data.strides)}\n"
            f"  Batch size: {format_value('data.batch_size', config.data.batch_size)}\n"
            f"  Enable Augmentation: {format_value('data.enable_augmentation', config.data.enable_augmentation)}\n\n"  # ✅ Added this line
            f"[yellow]Training Settings:[/]\n"
            f"  Model Type: {format_value('model.type', config.model.type)}\n"
            f"  Max Epochs: {format_value('training.max_epochs', config.training.max_epochs)}\n"
            f"  Learning Rate: {format_value('training.learning_rate', config.training.learning_rate)}\n"
            f"  Weight Decay: {format_value('training.weight_decay', config.training.weight_decay)}\n"
            f"  Early Stopping Patience: {format_value('training.early_stopping_patience', config.training.early_stopping_patience)}\n"
            f"  Gradient Clip Value: {format_value('training.gradient_clip_val', config.training.gradient_clip_val)}",
            title="Configuration",
            style="blue"
        ))