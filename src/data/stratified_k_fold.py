from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd
from rich import print as rprint
import yaml


class StratifiedSubjectWindowKFold:
    def __init__(self, n_splits: int = 5, test_size: float = 0.1, shuffle: bool = True, random_state: int = None, yaml_split_path: str = None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.window_size = None
        self.stride = None
        self.verbose = False
        self.yaml_split_path = yaml_split_path
        self.yaml_splits = self._load_yaml_splits() if yaml_split_path else None
        
    
    def _load_yaml_splits(self):
        """Load splits from YAML file if it exists."""
        try:
            if Path(self.yaml_split_path).exists():
                with open(self.yaml_split_path, 'r') as f:
                    return yaml.safe_load(f)
            return None
        except Exception as e:
            if self.verbose:
                rprint(f"[yellow]Warning: Could not load YAML splits: {e}. Using random splits.[/yellow]")
            return None
        
    def _get_fold_splits(self, fold):
        """Get train/val/test splits for a specific fold from YAML."""
        if not self.yaml_splits:
            return None
            
        fold_key = f"Fold{fold + 1}"
        if fold_key not in self.yaml_splits:
            raise ValueError(f"Fold {fold + 1} not found in YAML splits configuration")
        
        fold_data = self.yaml_splits[fold_key]
        return (
            fold_data.get('Train', []),
            fold_data.get('Val', []),
            fold_data.get('Test', [])
        )
        
        
    def get_subject_label(self, subject_data: pd.DataFrame, label_col: str) -> int:
        """Get the label for a subject (most frequent label)"""
        return subject_data[label_col].mode()[0]

    def get_subject_info(self, data: pd.DataFrame, column_names: Dict[str, str]) -> Dict:
        """Compute information about each subject's data"""
        subject_info = {}

        for subject in data.index.get_level_values(0).unique():
            subject_data = data.loc[subject]
            label = self.get_subject_label(subject_data, column_names['label'])

            subject_info[subject] = {
                'label': label,
                'n_segments': len(subject_data),
                'label_distribution': subject_data[column_names['label']].value_counts().to_dict(),
                'task_counts': subject_data[column_names['task']].value_counts().to_dict()
            }

        return subject_info

    def split(self, data: pd.DataFrame, column_names: Dict[str, str]) -> List[Tuple[List[int], List[int], List[int]]]:
        """
        Generate indices to split data into training, validation, and test sets.
        Returns a list of (train_idx, val_idx, test_idx) tuples for each fold.
        """
        if self.window_size is None or self.stride is None:
            raise ValueError("Must call set_window_params before split")

        splits = []
        
        # If using YAML splits
        if self.yaml_splits:
            # Process all folds
            for fold in range(self.n_splits):
                yaml_split = self._get_fold_splits(fold)
                if yaml_split:
                    train_subjects, val_subjects, test_subjects = yaml_split
                    
                    # Create masks for each split
                    train_mask = data.index.get_level_values(0).isin(train_subjects)
                    val_mask = data.index.get_level_values(0).isin(val_subjects)
                    test_mask = data.index.get_level_values(0).isin(test_subjects)
                    
                    # Get indices
                    train_indices = np.where(train_mask)[0]
                    val_indices = np.where(val_mask)[0]
                    test_indices = np.where(test_mask)[0]
                    
                    if self.verbose:
                        self._print_split_stats(fold, data, train_subjects, val_subjects, test_subjects, 
                                            self.get_subject_info(data, column_names))
                    
                    splits.append((train_indices, val_indices, test_indices))
            
        else:
            np.random.seed(self.random_state)
            # Get subject information
            subject_info = self.get_subject_info(data, column_names)

            # Prepare data for stratification
            subjects = np.array(list(subject_info.keys()))
            labels = np.array([info['label'] for info in subject_info.values()])

            # First, split out the test set
            test_size_n = int(len(subjects) * self.test_size)

            # Stratified split for test set
            sss = StratifiedKFold(n_splits=int(1 / self.test_size), shuffle=True, random_state=self.random_state)
            train_val_idx, test_idx = next(sss.split(subjects, labels))

            # Get test subjects and remaining subjects for train/val
            test_subjects = subjects[test_idx]
            train_val_subjects = subjects[train_val_idx]
            train_val_labels = labels[train_val_idx]

            # Now split remaining data into k folds
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            splits = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_subjects, train_val_labels)):
                # Get subject IDs for this split
                train_subjects = train_val_subjects[train_idx]
                val_subjects = train_val_subjects[val_idx]

                # Get data indices for each split
                train_mask = data.index.get_level_values(0).isin(train_subjects)
                val_mask = data.index.get_level_values(0).isin(val_subjects)
                test_mask = data.index.get_level_values(0).isin(test_subjects)

                train_indices = np.where(train_mask)[0]
                val_indices = np.where(val_mask)[0]
                test_indices = np.where(test_mask)[0]

                if self.verbose:
                    self._print_split_stats(fold, data, train_subjects, val_subjects, test_subjects, subject_info)

                splits.append((train_indices, val_indices, test_indices))

        return splits

    def _print_split_stats(self, fold: int, data: pd.DataFrame, train_subjects: np.ndarray,
                           val_subjects: np.ndarray, test_subjects: np.ndarray, subject_info: Dict):
        """Print statistics for a fold"""

        def get_split_stats(subjects):
            n_segments = sum(subject_info[s]['n_segments'] for s in subjects)
            n_pos = sum(1 for s in subjects if subject_info[s]['label'] == 1)
            n_neg = len(subjects) - n_pos
            return {
                'n_subjects': len(subjects),
                'n_segments': n_segments,
                'n_pos': n_pos,
                'n_neg': n_neg
            }

        train_stats = get_split_stats(train_subjects)
        val_stats = get_split_stats(val_subjects)
        test_stats = get_split_stats(test_subjects)

        rprint(f"\n[bold cyan]Fold {fold + 1} Statistics:[/bold cyan]")

        rprint(f"\n[green]Training Set:[/green]")
        rprint(
            f"Subjects: {train_stats['n_subjects']} ({train_stats['n_pos']} positive, {train_stats['n_neg']} negative)")
        rprint(f"Total Segments: {train_stats['n_segments']}")

        rprint(f"\n[yellow]Validation Set:[/yellow]")
        rprint(f"Subjects: {val_stats['n_subjects']} ({val_stats['n_pos']} positive, {val_stats['n_neg']} negative)")
        rprint(f"Total Segments: {val_stats['n_segments']}")

        rprint(f"\n[blue]Test Set:[/blue]")
        rprint(f"Subjects: {test_stats['n_subjects']} ({test_stats['n_pos']} positive, {test_stats['n_neg']} negative)")
        rprint(f"Total Segments: {test_stats['n_segments']}")

    def set_window_params(self, window_size: int, stride: int, verbose: bool = False):
        """Set window parameters for splitting"""
        self.window_size = window_size
        self.stride = stride
        self.verbose = verbose
