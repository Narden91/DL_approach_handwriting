import torch
import numpy as np
from rich import print as rprint
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class GradientModelExplainer:
    def __init__(
        self, 
        model, 
        datamodule, 
        feature_names: List[str], 
        verbose: bool = True,
        n_samples: int = 100,
        batch_size: int = 32
    ):
        """
        Initialize the gradient-based model explainer.
        
        Args:
            model: The trained PyTorch model
            datamodule: Lightning DataModule containing the data
            feature_names: List of feature names
            verbose: Whether to print detailed information
            n_samples: Number of samples to use for importance calculation
            batch_size: Batch size for processing
        """
        self.model = model
        self.datamodule = datamodule
        self.feature_names = feature_names
        self.verbose = verbose
        self.device = next(model.parameters()).device
        self.n_samples = n_samples
        self.batch_size = batch_size

    def _compute_integrated_gradients(
        self, 
        features: torch.Tensor,
        task_ids: torch.Tensor,
        masks: torch.Tensor,
        steps: int = 50
    ) -> torch.Tensor:
        """
        Compute integrated gradients for feature attribution.
        
        This method implements the integrated gradients algorithm, which provides
        a way to attribute the prediction of a deep network to its input features.
        """
        # Create a baseline (zeros) and scale features
        baseline = torch.zeros_like(features)
        scaled_features = [baseline + (float(i) / steps) * (features - baseline) 
                         for i in range(steps + 1)]
        scaled_features = torch.cat(scaled_features, dim=0)
        
        # Expand task_ids and masks to match the scaled features
        expanded_task_ids = task_ids.repeat(steps + 1, 1)
        expanded_masks = masks.repeat(steps + 1, 1)
        
        # Enable gradient computation
        scaled_features.requires_grad_(True)
        
        # Compute gradients in batches
        gradients = []
        for i in range(0, len(scaled_features), self.batch_size):
            batch_features = scaled_features[i:i + self.batch_size]
            batch_task_ids = expanded_task_ids[i:i + self.batch_size]
            batch_masks = expanded_masks[i:i + self.batch_size]
            
            # Forward pass
            outputs = self.model(batch_features, batch_task_ids, batch_masks)
            
            # Compute gradients
            grad = torch.autograd.grad(
                outputs=outputs.sum(),
                inputs=batch_features,
                create_graph=False,
                retain_graph=False
            )[0]
            gradients.append(grad)
        
        # Concatenate all gradients
        gradients = torch.cat(gradients, dim=0)
        
        # Compute mean gradients across steps
        avg_gradients = torch.mean(gradients.view(steps + 1, *features.shape), dim=0)
        
        # Compute final attribution
        integrated_gradients = (features - baseline) * avg_gradients
        return integrated_gradients

    def analyze_feature_importance(self, fold: int) -> Dict[str, float]:
        """
        Calculate feature importance using integrated gradients.
        
        This method computes feature importance by analyzing how each feature
        contributes to the model's predictions using integrated gradients.
        """
        if self.verbose:
            rprint(f"[blue]Analyzing feature importance for fold {fold + 1}...[/blue]")
            
        try:
            # Get test samples
            test_loader = self.datamodule.test_dataloader()
            all_attributions = []
            samples_processed = 0
            
            self.model.eval()
            for batch in test_loader:
                features, _, task_ids, masks = [x.to(self.device) for x in batch]
                
                # Compute integrated gradients for the batch
                attributions = self._compute_integrated_gradients(
                    features, 
                    task_ids, 
                    masks
                )
                
                # Store the absolute attributions
                all_attributions.append(attributions.abs().cpu().detach())
                
                samples_processed += len(features)
                if samples_processed >= self.n_samples:
                    break
            
            # Concatenate all attributions
            all_attributions = torch.cat(all_attributions, dim=0)
            
            # Average attributions across samples and sequence length
            feature_importance = all_attributions.mean(dim=(0, 1)).numpy()
            
            # Normalize importance scores
            feature_importance = feature_importance / feature_importance.sum()
            
            # Create importance dictionary
            importance_dict = {
                name: float(importance)
                for name, importance in zip(self.feature_names, feature_importance)
            }
            
            if self.verbose:
                rprint(f"[green]Successfully calculated feature importance for fold {fold + 1}[/green]")
                self._print_top_features(importance_dict)
                
            return importance_dict
            
        except Exception as e:
            if self.verbose:
                rprint(f"[red]Error in analyze_feature_importance: {str(e)}[/red]")
            raise

    def _print_top_features(self, importance_dict: Dict[str, float], top_n: int = 10):
        """Print the top N most important features."""
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        rprint("\n[yellow]Top Feature Importance:[/yellow]")
        for feature, importance in sorted_features[:top_n]:
            rprint(f"{feature}: {importance:.4f}")

    def analyze_task_importance(self, fold: int) -> Dict[int, float]:
        """
        Calculate task importance using gradient-based attribution.
        
        This method analyzes how different tasks contribute to the model's predictions
        by examining the gradients with respect to the task embeddings.
        """
        if self.verbose:
            rprint(f"[blue]Analyzing task importance for fold {fold + 1}...[/blue]")
            
        task_importance = {}
        activation_counts = {}
        
        try:
            test_loader = self.datamodule.test_dataloader()
            
            self.model.eval()
            for batch in test_loader:
                features, labels, task_ids, masks = [x.to(self.device) for x in batch]
                
                # Enable gradient computation for features
                features.requires_grad_(True)
                
                # Forward pass
                outputs = self.model(features, task_ids, masks)
                
                # Compute gradients
                gradients = torch.autograd.grad(
                    outputs=outputs.sum(),
                    inputs=features,
                    create_graph=False,
                    retain_graph=False
                )[0]
                
                # Compute feature attributions for each task
                attributions = (features * gradients).abs().sum(dim=-1).mean(dim=-1)
                
                # Aggregate by task
                for i, task_id in enumerate(task_ids):
                    task_id = task_id.item()
                    if task_id not in task_importance:
                        task_importance[task_id] = 0.0
                        activation_counts[task_id] = 0
                    
                    task_importance[task_id] += attributions[i].item()
                    activation_counts[task_id] += 1
                
                if sum(activation_counts.values()) >= self.n_samples:
                    break
            
            # Average and normalize importance scores
            for task_id in task_importance:
                if activation_counts[task_id] > 0:
                    task_importance[task_id] /= activation_counts[task_id]
            
            # Normalize scores
            max_importance = max(task_importance.values())
            if max_importance > 0:
                task_importance = {
                    task: score / max_importance
                    for task, score in task_importance.items()
                }
            
            return task_importance
            
        except Exception as e:
            if self.verbose:
                rprint(f"[red]Error in analyze_task_importance: {str(e)}[/red]")
            raise

    @staticmethod
    def aggregate_importances(fold_importances: List[Dict]) -> Dict[str, float]:
        """
        Aggregate importance scores across folds with robust statistics.
        """
        if not fold_importances:
            return {}
            
        all_keys = set().union(*fold_importances)
        aggregated = {}
        
        for key in all_keys:
            values = [d.get(key, 0.0) for d in fold_importances]
            
            # Remove outliers using IQR method
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            valid_values = [
                v for v in values 
                if (v >= q1 - 1.5 * iqr) and (v <= q3 + 1.5 * iqr)
            ]
            
            aggregated[key] = float(np.mean(valid_values) if valid_values else 0.0)
            
        # Normalize final scores
        total = sum(aggregated.values())
        if total > 0:
            aggregated = {k: v / total for k, v in aggregated.items()}
            
        return aggregated

    def plot_feature_importance(
        self, 
        importance_dict: Dict[str, float], 
        title: str = "Feature Importance",
        top_n: Optional[int] = None,
        figure_size: Tuple[int, int] = (12, 6)
    ):
        """
        Create a visualization of feature importance scores.
        """
        if not importance_dict:
            if self.verbose:
                rprint("[yellow]No feature importance data to plot[/yellow]")
            return
            
        # Sort features by importance
        sorted_items = sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Take top N features if specified
        if top_n:
            sorted_items = sorted_items[:top_n]
            
        features, scores = zip(*sorted_items)
        
        plt.figure(figsize=figure_size)
        bars = plt.bar(features, scores)
        plt.xticks(rotation=45, ha='right')
        plt.title(title)
        plt.ylabel('Importance Score')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.4f}',
                ha='center',
                va='bottom'
            )
            
        plt.tight_layout()
        plt.show()
        
    def plot_task_importance(
        self, 
        importance_dict: Dict[int, float], 
        title: str = "Task Importance",
        top_n: Optional[int] = None,
        figure_size: Tuple[int, int] = (12, 6),
        color_map: str = 'viridis'
    ):
        """
        Create a visualization of task importance scores.
        
        Args:
            importance_dict: Dictionary mapping task IDs to their importance scores
            title: Title for the plot
            top_n: Optional number of top tasks to show
            figure_size: Tuple specifying figure dimensions
            color_map: Colormap to use for the bars
        """
        if not importance_dict:
            if self.verbose:
                rprint("[yellow]No task importance data to plot[/yellow]")
            return
            
        # Sort tasks by importance
        sorted_items = sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Take top N tasks if specified
        if top_n:
            sorted_items = sorted_items[:top_n]
            
        tasks, scores = zip(*sorted_items)
        
        # Create color gradient
        colors = plt.cm.get_cmap(color_map)(np.linspace(0, 0.8, len(tasks)))
        
        plt.figure(figsize=figure_size)
        bars = plt.bar([f"Task {t}" for t in tasks], scores, color=colors)
        
        # Customize the plot
        plt.xticks(rotation=45, ha='right')
        plt.title(title, pad=20)
        plt.ylabel('Importance Score')
        plt.xlabel('Task ID')
        
        # Add grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.4f}',
                ha='center',
                va='bottom',
                fontsize=8
            )
            
        # Add a brief explanation of the visualization
        plt.figtext(
            0.02, 
            0.02, 
            'Higher scores indicate tasks that have more influence on the model\'s predictions',
            style='italic', 
            fontsize=8
        )
        
        plt.tight_layout()
        plt.show()
        
        if self.verbose:
            # Print numerical summary
            rprint("\n[yellow]Task Importance Summary:[/yellow]")
            for task, score in sorted_items:
                rprint(f"Task {task}: {score:.4f}")