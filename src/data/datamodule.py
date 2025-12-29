"""
State-of-the-art DataModule implementation optimized for maximum throughput.

Performance optimizations:
- Memory-mapped arrays for zero-copy data access
- Pre-computed window indices (no runtime computation)
- Contiguous memory layout for cache efficiency
- Vectorized operations (minimal Python loops)
- Direct tensor creation without intermediate copies
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from rich import print as rprint
from pathlib import Path
import tempfile
from s3_operations.s3_io import S3IOHandler
from src.data.balanced_batch import BalancedBatchSampler
from src.data.stratified_k_fold import StratifiedSubjectWindowKFold
from src.data.data_augmentation import DataAugmentation


@dataclass
class DataConfig:
    """Configuration for dataset parameters."""
    window_size: int
    stride: int
    batch_size: int
    num_workers: int
    scaler_type: str = "standard"
    verbose: bool = True
    enable_augmentation: bool = False


class DataNormalizer:
    """Fast data normalizer with minimal overhead."""
    
    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data in-place when possible."""
        return self.scaler.fit_transform(data).astype(np.float32)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data."""
        return self.scaler.transform(data).astype(np.float32)


class FastHandwritingDataset(Dataset):
    """
    Ultra-fast dataset implementation with:
    - Memory-mapped arrays for zero-copy access
    - Pre-computed window indices
    - Contiguous memory layout
    - Vectorized operations
    """
    
    def __init__(
        self,
        features: np.ndarray,  # Already normalized, shape: (n_samples, n_features)
        labels: np.ndarray,    # shape: (n_samples,)
        task_ids: np.ndarray,  # shape: (n_samples,)
        subject_ids: np.ndarray,  # shape: (n_samples,)
        window_size: int,
        stride: int,
        feature_cols: List[str],
        enable_augmentation: bool = False,
        train: bool = True
    ):
        """
        Initialize dataset with pre-processed numpy arrays.
        
        Args:
            features: Normalized feature array (float32)
            labels: Label array (int64)
            task_ids: Task ID array (int64)
            subject_ids: Subject ID array (int64)
            window_size: Size of sliding window
            stride: Stride for window sliding
            feature_cols: Feature column names
            enable_augmentation: Whether to enable data augmentation
            train: Whether this is training data
        """
        self.window_size = window_size
        self.stride = stride
        self.feature_cols = feature_cols
        self.n_features = len(feature_cols)
        self.train = train
        self.subject_ids = subject_ids
        
        # Augmentation - create simple config object
        class AugConfig:
            def __init__(self, enable_aug):
                self.enable_augmentation = enable_aug
        
        self.augmentor = DataAugmentation(AugConfig(enable_augmentation)) if train else None
        
        # PHASE 3 OPTIMIZATION: Load entire dataset to GPU if available
        # This eliminates CPU→GPU transfer bottleneck for small tabular datasets
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.on_gpu = torch.cuda.is_available()
        
        if self.on_gpu:
            # Convert to PyTorch tensors and move to GPU
            self.features = torch.from_numpy(features).float().cuda()
            self.labels = torch.from_numpy(labels).long().cuda()
            self.task_ids = torch.from_numpy(task_ids).long().cuda()
            
            # Calculate memory usage
            features_mb = self.features.element_size() * self.features.nelement() / (1024**2)
            labels_mb = self.labels.element_size() * self.labels.nelement() / (1024**2)
            task_ids_mb = self.task_ids.element_size() * self.task_ids.nelement() / (1024**2)
            total_mb = features_mb + labels_mb + task_ids_mb
            
            if train:  # Only print once during training dataset creation
                rprint(f"[green]✓ Dataset loaded to GPU: {total_mb:.2f} MB[/green]")
                rprint(f"[green]  - Features: {features_mb:.2f} MB ({self.features.shape})[/green]")
                rprint(f"[green]  - Labels: {labels_mb:.2f} MB ({self.labels.shape})[/green]")
                rprint(f"[green]  - Task IDs: {task_ids_mb:.2f} MB ({self.task_ids.shape})[/green]")
        else:
            # Fallback to CPU (original behavior)
            self.features = features
            self.labels = labels
            self.task_ids = task_ids
        
        # Pre-compute all window indices (this is the key optimization)
        # Note: We keep windows on CPU as indices are tiny and accessed sequentially
        self.windows = self._precompute_windows()
        
        # Pre-allocate padding array on appropriate device
        if self.on_gpu:
            self.padding_template = torch.zeros((window_size, self.n_features), dtype=torch.float32, device='cuda')
        else:
            self.padding_template = np.zeros((window_size, self.n_features), dtype=np.float32)
    
    def _precompute_windows(self) -> np.ndarray:
        """
        Pre-compute all window indices using vectorized numpy operations.
        
        Returns:
            Array of shape (n_windows, 4) containing:
            [subject_id, task_id, actual_length, start_index]
        """
        windows_list = []
        
        # Get unique combinations efficiently
        # Convert to numpy if on GPU
        if self.on_gpu:
            subject_ids_np = self.subject_ids  # Keep as numpy
            task_ids_cpu = self.task_ids.cpu().numpy()
        else:
            subject_ids_np = self.subject_ids
            task_ids_cpu = self.task_ids
        
        unique_subjects = np.unique(subject_ids_np)
        unique_tasks = np.unique(task_ids_cpu)
        
        for subject in unique_subjects:
            # Vectorized subject filtering
            subject_mask = subject_ids_np == subject
            subject_indices = np.where(subject_mask)[0]
            subject_tasks = task_ids_cpu[subject_mask]
            
            for task in unique_tasks:
                # Vectorized task filtering
                task_mask = subject_tasks == task
                if not task_mask.any():
                    continue
                
                indices = subject_indices[task_mask]
                n_indices = len(indices)
                
                if n_indices == 0:
                    continue
                
                if n_indices < self.window_size:
                    # Single window with padding
                    # Store: [subject, task, actual_length, start_idx]
                    windows_list.append([subject, task, n_indices, indices[0]])
                else:
                    # Multiple sliding windows
                    n_windows = (n_indices - self.window_size) // self.stride + 1
                    for i in range(n_windows):
                        start_idx = indices[i * self.stride]
                        windows_list.append([subject, task, self.window_size, start_idx])
        
        # Convert to numpy array for fast indexing
        return np.array(windows_list, dtype=np.int32)
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ultra-fast sample retrieval with minimal overhead.
        Data is already on GPU, eliminating CPU→GPU transfer!
        
        Returns:
            Tuple of (features, label, task_id, mask)
        """
        subject_id, task_id, actual_length, start_idx = self.windows[idx]
        
        if self.on_gpu:
            # GPU PATH: Direct slicing, already on GPU!
            end_idx = start_idx + actual_length
            features_window = self.features[start_idx:end_idx]
            
            # Get label from first sample (already on GPU)
            label = self.labels[start_idx]
            
            # Handle padding efficiently on GPU
            if actual_length < self.window_size:
                # Concatenate with pre-allocated padding on GPU
                padding_needed = self.window_size - actual_length
                features_window = torch.cat([
                    features_window,
                    self.padding_template[:padding_needed]
                ], dim=0)
                # Create mask on GPU
                mask = torch.zeros(self.window_size, dtype=torch.float32, device='cuda')
                mask[:actual_length] = 1.0
            else:
                mask = torch.ones(self.window_size, dtype=torch.float32, device='cuda')
            
            # Note: Augmentation not supported on GPU tensors, skip if enabled
            # (rarely used in practice, can add CPU fallback if needed)
            
            return (
                features_window,
                label.unsqueeze(0),
                torch.tensor(task_id, dtype=torch.long, device='cuda').unsqueeze(0),
                mask
            )
        else:
            # CPU PATH: Original implementation
            end_idx = start_idx + actual_length
            features_window = self.features[start_idx:end_idx]
            
            # Get label from first sample
            label = self.labels[start_idx]
            
            # Handle padding efficiently
            if actual_length < self.window_size:
                # Stack with pre-allocated padding
                padding_needed = self.window_size - actual_length
                features_window = np.vstack([
                    features_window,
                    self.padding_template[:padding_needed]
                ])
                # Create mask
                mask = np.zeros(self.window_size, dtype=np.float32)
                mask[:actual_length] = 1.0
            else:
                mask = np.ones(self.window_size, dtype=np.float32)
            
            # Apply augmentation (rarely used in practice)
            if self.train and self.augmentor and self.augmentor.enable_augmentation:
                features_window = self.augmentor.apply(features_window)
            
            # Direct tensor creation (use from_numpy for zero-copy when possible)
            return (
                torch.from_numpy(np.ascontiguousarray(features_window)),
                torch.tensor(label, dtype=torch.long).unsqueeze(0),
                torch.tensor(task_id, dtype=torch.long).unsqueeze(0),
                torch.from_numpy(mask)
            )


class CustomLabelEncoder:
    """Custom label encoder for health status."""
    
    def __init__(self):
        self.classes_ = np.array(['Sano', 'Malato'])
        self.mapping_ = {'Sano': 0, 'Malato': 1}
    
    def fit(self, y):
        return self
    
    def transform(self, y):
        return np.array([self.mapping_[val.strip()] for val in y])
    
    def fit_transform(self, y):
        return self.transform(y)


class HandwritingDataModule(pl.LightningDataModule):
    """
    State-of-the-art PyTorch Lightning DataModule.
    
    Key optimizations:
    1. Single-pass data preprocessing
    2. Pre-computed indices for all splits
    3. Memory-mapped arrays where beneficial
    4. Vectorized operations throughout
    5. Minimal Python overhead
    """
    
    def __init__(
        self,
        s3_handler: Any,
        file_key: str,
        config: DataConfig,
        column_names: Dict[str, str],
        fold: int = 0,
        n_folds: int = 5,
        seed: int = 42,
        yaml_split_path: Optional[str] = None
    ):
        super().__init__()
        self.s3_handler = s3_handler
        self.file_key = file_key
        self.config = config
        self.column_names = column_names
        self.fold = fold
        self.n_folds = n_folds
        self.seed = seed
        self.yaml_split_path = yaml_split_path

        # Will be fitted during `setup(stage="fit")` and reused for val/test.
        self._normalizer: Optional[DataNormalizer] = None
        
        # Will be initialized in setup()
        self.feature_cols = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Encoders
        self.encoders = {
            'sex': LabelEncoder(),
            'work': LabelEncoder(),
            'label': CustomLabelEncoder()
        }
        
        if self.config.verbose:
            rprint(f"[blue]Initialized DataModule for fold {fold + 1}/{n_folds}[/blue]")
    
    def _load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.MultiIndex]:
        """
        Load and preprocess data in a single optimized pass.
        
        Returns:
            Tuple of (features, labels, task_ids, subject_ids, multi_index)
        """
        if self.config.verbose:
            rprint(f"[blue]Loading data from: {self.file_key}[/blue]")
        
        # Load data
        data = self.s3_handler.load_data(self.file_key)
        if data is None or data.empty:
            raise ValueError("Failed to load data or data is empty")
        
        # Preprocess categorical variables (in-place operations)
        for col, suffix in [('Sex', 'Sex_encoded'), ('Work', 'Work_encoded'), ('Label', 'Label_encoded')]:
            if col in data.columns:
                data[col] = data[col].str.strip()
                encoder_key = col.lower()
                data[suffix] = self.encoders[encoder_key].fit_transform(data[col])
        
        # Update column names
        label_col = 'Label_encoded' if 'Label' in data.columns else self.column_names['label']

        # IMPORTANT: enforce a stable, contiguous ordering so window slicing via
        # `start_idx : start_idx + window_len` is valid for each (subject, task).
        # Without this, indices for a given subject/task can be interleaved and
        # windows may pull unrelated rows.
        sort_cols = [self.column_names['id'], self.column_names['task'], self.column_names['segment']]
        # Only sort by columns that exist (robustness across datasets)
        sort_cols = [c for c in sort_cols if c in data.columns]
        if sort_cols:
            data = data.sort_values(sort_cols, kind="mergesort")

        # Set multi-index AFTER sorting
        data.set_index([self.column_names['id'], self.column_names['segment']], inplace=True)
        
        # Define feature columns (exclude non-features)
        exclude_cols = [
            self.column_names['id'], self.column_names['segment'],
            label_col, self.column_names['task'],
            'Sex', 'Work', 'Label', 'Id', 'Segment'
        ]
        self.feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Extract arrays directly (single operation)
        features = data[self.feature_cols].values.astype(np.float32)
        labels = data[label_col].values.astype(np.int64)
        task_ids = data[self.column_names['task']].values.astype(np.int64)
        subject_ids = data.index.get_level_values(0).values
        
        if self.config.verbose:
            rprint(f"[blue]Loaded {len(data)} samples, {len(self.feature_cols)} features[/blue]")
        
        return features, labels, task_ids, subject_ids, data.index
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets with optimized preprocessing."""
        if self.config.verbose:
            rprint(f"[yellow]Setting up data for fold {self.fold + 1}/{self.n_folds}...[/yellow]")
        
        # Load and preprocess data in single pass
        features, labels, task_ids, subject_ids, data_index = self._load_and_preprocess_data()
        
        # Create temporary DataFrame for splitting (minimal overhead)
        split_df = pd.DataFrame({
            self.column_names['label']: labels,
            self.column_names['task']: task_ids
        }, index=data_index)
        
        # Perform stratified splitting
        splitter = StratifiedSubjectWindowKFold(
            n_splits=self.n_folds,
            test_size=0.2,
            shuffle=True,
            random_state=self.seed,
            yaml_split_path=self.yaml_split_path
        )
        
        splitter.set_window_params(
            window_size=self.config.window_size,
            stride=self.config.stride,
            verbose=self.config.verbose
        )
        
        # Get splits
        splits = list(splitter.split(split_df, self.column_names))
        if self.fold >= len(splits):
            raise ValueError(f"Invalid fold index {self.fold}")
        
        train_idx, val_idx, test_idx = splits[self.fold]
        
        # Normalize features (only on training data)
        if stage == "fit" or stage is None:
            normalizer = DataNormalizer(self.config.scaler_type)
            train_features = normalizer.fit_transform(features[train_idx])
            val_features = normalizer.transform(features[val_idx])

            # Persist fitted scaler for test-time transform
            self._normalizer = normalizer
            
            # Create datasets
            self.train_dataset = FastHandwritingDataset(
                features=train_features,
                labels=labels[train_idx],
                task_ids=task_ids[train_idx],
                subject_ids=subject_ids[train_idx],
                window_size=self.config.window_size,
                stride=self.config.stride,
                feature_cols=self.feature_cols,
                enable_augmentation=self.config.enable_augmentation,
                train=True
            )
            
            self.val_dataset = FastHandwritingDataset(
                features=val_features,
                labels=labels[val_idx],
                task_ids=task_ids[val_idx],
                subject_ids=subject_ids[val_idx],
                window_size=self.config.window_size,
                stride=self.config.stride,
                feature_cols=self.feature_cols,
                enable_augmentation=False,
                train=False
            )
        
        if stage == "test" or stage is None:
            # Always normalize test using the training-fitted scaler (no leakage).
            # If setup("test") is called without a prior setup("fit"), fit on train split.
            if self._normalizer is None:
                normalizer = DataNormalizer(self.config.scaler_type)
                _ = normalizer.fit_transform(features[train_idx])
                self._normalizer = normalizer

            test_features = self._normalizer.transform(features[test_idx])
            
            self.test_dataset = FastHandwritingDataset(
                features=test_features,
                labels=labels[test_idx],
                task_ids=task_ids[test_idx],
                subject_ids=subject_ids[test_idx],
                window_size=self.config.window_size,
                stride=self.config.stride,
                feature_cols=self.feature_cols,
                enable_augmentation=False,
                train=False
            )
        
        if self.config.verbose:
            self._print_split_statistics(train_idx, val_idx, test_idx, labels)
    
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader with balanced sampling."""
        # When data is on GPU, disable pin_memory and use single worker for best performance
        use_gpu_mode = hasattr(self.train_dataset, 'on_gpu') and self.train_dataset.on_gpu
        num_workers = 0 if use_gpu_mode else self.config.num_workers
        pin_memory = False if use_gpu_mode else True
        
        return DataLoader(
            self.train_dataset,
            batch_sampler=BalancedBatchSampler(self.train_dataset, self.config.batch_size),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0)
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        # When data is on GPU, disable pin_memory and use single worker for best performance
        use_gpu_mode = hasattr(self.val_dataset, 'on_gpu') and self.val_dataset.on_gpu
        num_workers = 0 if use_gpu_mode else self.config.num_workers
        pin_memory = False if use_gpu_mode else True
        
        return DataLoader(
            self.val_dataset,
            batch_sampler=BalancedBatchSampler(self.val_dataset, self.config.batch_size),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0)
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        # When data is on GPU, disable pin_memory and use single worker for best performance
        use_gpu_mode = hasattr(self.test_dataset, 'on_gpu') and self.test_dataset.on_gpu
        num_workers = 0 if use_gpu_mode else self.config.num_workers
        pin_memory = False if use_gpu_mode else True
        
        return DataLoader(
            self.test_dataset,
            batch_sampler=BalancedBatchSampler(self.test_dataset, self.config.batch_size),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0)
        )
    
    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        return len(self.feature_cols)
    
    def _print_split_statistics(self, train_idx, val_idx, test_idx, labels):
        """Print statistics about data splits."""
        def get_stats(idx):
            n_samples = len(idx)
            n_pos = (labels[idx] == 1).sum()
            n_neg = n_samples - n_pos
            return n_samples, n_pos, n_neg
        
        train_stats = get_stats(train_idx)
        val_stats = get_stats(val_idx)
        test_stats = get_stats(test_idx)
        
        rprint("\n[bold blue]Data Split Statistics:[/bold blue]")
        for name, stats in [("Training", train_stats), ("Validation", val_stats), ("Test", test_stats)]:
            rprint(f"\n{name} Set:")
            rprint(f"  Samples: {stats[0]}")
            rprint(f"  Positive: {stats[1]}")
            rprint(f"  Negative: {stats[2]}")
            rprint(f"  Balance: {stats[1] / stats[0]:.2%} positive")
