import os
from typing import List, Dict, Any
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint
from rich.panel import Panel

class ConfigOperations:
    @staticmethod
    def parse_list_param(param_str: str) -> List[int]:
        """Parse comma-separated string into list of integers."""
        try:
            return [int(x.strip()) for x in param_str.split(',')]
        except ValueError:
            raise ValueError("Parameters must be comma-separated integers")
    
    @staticmethod
    def parse_bool(value: str) -> bool:
        """Parse string to boolean."""
        return value.lower() in ('true', '1', 'yes', 'on')

    @staticmethod
    def get_env_config() -> Dict[str, Any]:
        """Get configuration from environment variables."""
        env_config = {}
        
        # Training parameters
        if max_epochs := os.getenv('MAX_EPOCHS'):
            try:
                env_config['training.max_epochs'] = int(max_epochs)
            except ValueError:
                rprint("[yellow]Warning: Invalid MAX_EPOCHS format[/yellow]")

        if learning_rate := os.getenv('LEARNING_RATE'):
            try:
                env_config['training.learning_rate'] = float(learning_rate)
            except ValueError:
                rprint("[yellow]Warning: Invalid LEARNING_RATE format[/yellow]")

        if weight_decay := os.getenv('WEIGHT_DECAY'):
            try:
                env_config['training.weight_decay'] = float(weight_decay)
            except ValueError:
                rprint("[yellow]Warning: Invalid WEIGHT_DECAY format[/yellow]")

        if early_stopping_patience := os.getenv('EARLY_STOPPING_PATIENCE'):
            try:
                env_config['training.early_stopping_patience'] = int(early_stopping_patience)
            except ValueError:
                rprint("[yellow]Warning: Invalid EARLY_STOPPING_PATIENCE format[/yellow]")

        if gradient_clip_val := os.getenv('GRADIENT_CLIP_VAL'):
            try:
                env_config['training.gradient_clip_val'] = float(gradient_clip_val)
            except ValueError:
                rprint("[yellow]Warning: Invalid GRADIENT_CLIP_VAL format[/yellow]")
        
        # Model parameter
        if model_name := os.getenv('MODEL_TYPE'):
            try:
                env_config['model_type'] = model_name
            except ValueError:
                rprint("[yellow]Warning: Invalid MODEL_TYPE format[/yellow]")
        
        # Data parameters
        if window_sizes_str := os.getenv('WINDOW_SIZES'):
            try:
                env_config['data.window_sizes'] = ConfigOperations.parse_list_param(window_sizes_str)
            except ValueError:
                rprint("[yellow]Warning: Invalid WINDOW_SIZES format[/yellow]")

        if strides_str := os.getenv('STRIDES'):
            try:
                env_config['data.strides'] = ConfigOperations.parse_list_param(strides_str)
            except ValueError:
                rprint("[yellow]Warning: Invalid STRIDES format[/yellow]")

        # General parameters
        if seed := os.getenv('SEED'):
            try:
                env_config['seed'] = int(seed)
            except ValueError:
                rprint("[yellow]Warning: Invalid SEED format[/yellow]")

        if verbose := os.getenv('VERBOSE'):
            env_config['verbose'] = ConfigOperations.parse_bool(verbose)

        if num_folds := os.getenv('NUM_FOLDS'):
            try:
                env_config['num_folds'] = int(num_folds)
            except ValueError:
                rprint("[yellow]Warning: Invalid NUM_FOLDS format[/yellow]")

        if test_mode := os.getenv('TEST_MODE'):
            env_config['test_mode'] = ConfigOperations.parse_bool(test_mode)

        if exp_name := os.getenv('EXPERIMENT_NAME'):
            env_config['experiment_name'] = exp_name

        return env_config

    @staticmethod
    def merge_configurations(hydra_cfg: DictConfig) -> DictConfig:
        """Merge configurations from environment variables with Hydra config."""
        # Get environment variables
        env_config = ConfigOperations.get_env_config()
        
        # Create mutable configuration
        config = OmegaConf.to_container(hydra_cfg, resolve=True)
        
        # Update with environment variables
        for key, value in env_config.items():
            OmegaConf.update(config, key, value, merge=True)
        
        # Convert back to OmegaConf
        merged_config = OmegaConf.create(config)
        
        # Display final configuration
        ConfigOperations.display_configuration(merged_config)
        
        return merged_config

    @staticmethod
    def display_configuration(config: DictConfig) -> None:
        """Display the final configuration."""
        rprint(Panel(
            f"[bold green]Training Configuration[/]\n\n"
            f"[yellow]General Settings:[/]\n"
            f"  Experiment: {config.experiment_name}\n"
            f"  Seed: {config.seed}\n"
            f"  Verbose: {config.verbose}\n"
            f"  Num Folds: {config.num_folds}\n"
            f"  Test Mode: {config.test_mode}\n\n"
            f"[yellow]Data Settings:[/]\n"
            f"  Window sizes: {config.data.window_sizes}\n"
            f"  Strides: {config.data.strides}\n"
            f"  Batch size: {config.data.batch_size}\n\n"
            f"[yellow]Training Settings:[/]\n"
            f"  Max Epochs: {config.training.max_epochs}\n"
            f"  Learning Rate: {config.training.learning_rate}\n"
            f"  Weight Decay: {config.training.weight_decay}\n"
            f"  Early Stopping Patience: {config.training.early_stopping_patience}\n"
            f"  Gradient Clip Value: {config.training.gradient_clip_val}",
            title="Configuration",
            style="blue"
        ))