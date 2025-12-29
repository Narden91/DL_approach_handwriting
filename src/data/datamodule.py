from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from rich import print as rprint
from s3_operations.s3_io import S3IOHandler
from src.data.balanced_batch import BalancedBatchSampler
from src.data.stratified_k_fold import StratifiedSubjectWindowKFold
from src.data.data_augmentation import DataAugmentation



@dataclass
class DataConfig:
    """Configuration class for dataset parameters.
    
    This class encapsulates all the configuration parameters needed for
    data processing and windowing operations.
    
    Attributes:
        window_size: Size of the sliding window
        stride: Step size for window sliding
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        scaler_type: Type of feature scaling to apply ("standard" or "robust")
        verbose: Whether to print detailed information
    """
    window_size: int
    stride: int
    batch_size: int
    num_workers: int
    scaler_type: str = "standard"
    verbose: bool = True
    enable_augmentation: bool = False  

class DataNormalizer:
    """Handles data normalization and preprocessing operations.
    
    This class provides a unified interface for data normalization,
    handling both feature-wise and window-wise normalization.
    
    Attributes:
        scaler_type: Type of scaler to use ("standard" or "robust")
        scaler: The fitted scaler instance
        feature_means: Running means of features
        feature_stds: Running standard deviations of features
    """
    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
        self.feature_means = None
        self.feature_stds = None
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit the scaler and transform the data.
        
        Args:
            data: Input DataFrame to normalize
            
        Returns:
            Normalized DataFrame
        """
        normalized_data = pd.DataFrame(
            self.scaler.fit_transform(data),
            columns=data.columns,
            index=data.index
        )
        return normalized_data
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the fitted scaler.
        
        Args:
            data: Input DataFrame to normalize
            
        Returns:
            Normalized DataFrame
        """
        return pd.DataFrame(
            self.scaler.transform(data),
            columns=data.columns,
            index=data.index
        )

class HandwritingDataset(Dataset):
    """Dataset class for handwriting analysis.
    
    This class handles the creation and management of windowed data samples
    for handwriting analysis, including feature normalization and
    window creation.
    
    Attributes:
        data: The input DataFrame
        window_size: Size of sliding windows
        stride: Step size for window creation
        feature_cols: List of feature column names
        column_names: Dictionary mapping column types to names
        normalizer: DataNormalizer instance
        windows: List of created data windows
    """
    def __init__(
        self,
        data: pd.DataFrame,
        config: DataConfig,
        feature_cols: List[str],
        column_names: Dict[str, str],
        normalizer: Optional[DataNormalizer] = None,
        train: bool = True
    ):
        """Initialize the HandwritingDataset.
        
        Args:
            data: Input DataFrame
            config: DataConfig instance
            feature_cols: List of feature column names
            column_names: Dictionary mapping column types to names
            normalizer: Optional pre-fitted DataNormalizer
            train: Whether this is a training dataset
        """
        self.data = data.copy()
        self.window_size = config.window_size
        self.stride = config.stride
        self.feature_cols = feature_cols
        self.column_names = column_names
        self.train = train
        
        self.augmentor = DataAugmentation(config) if train else None

        # Initialize features and labels
        self.features_df = self.data[feature_cols]
        self.labels_df = self.data[self.column_names['label']]

        # Handle normalization
        if normalizer is None and train:
            self.normalizer = DataNormalizer(config.scaler_type)
            self.features_df = self.normalizer.fit_transform(self.features_df)
        elif normalizer is not None:
            self.normalizer = normalizer
            self.features_df = self.normalizer.transform(self.features_df)

        # Create windows
        self.windows = self._create_windows()
        
        if config.verbose:
            rprint(f"Created dataset with {len(self.windows)} windows")

    def _create_windows(self) -> List[Tuple[int, int, int, List[int]]]:
        """Create sliding windows for each subject and task combination.
        
        Returns:
            List of tuples containing (subject_id, task_id, window_size, indices)
        """
        windows = []
        subjects = self.data.index.get_level_values(0).unique()
        tasks = self.data[self.column_names['task']].unique()

        for subject in subjects:
            for task in tasks:
                mask = (self.data.index.get_level_values(0) == subject) & \
                       (self.data[self.column_names['task']] == task)
                indices = np.where(mask)[0]

                if len(indices) == 0:
                    continue

                if len(indices) < self.window_size:
                    windows.append((subject, task, len(indices), indices.tolist()))
                else:
                    for start_idx in range(0, len(indices) - self.window_size + 1, self.stride):
                        end_idx = start_idx + self.window_size
                        window_indices = indices[start_idx:end_idx].tolist()
                        windows.append((subject, task, self.window_size, window_indices))

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single data sample.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (features, label, task_id, mask)
        """
        subject_id, task_id, window_size, indices = self.windows[idx]

        # Handle missing values and extreme values
        features_window = np.nan_to_num(self.features_df.iloc[indices].values, nan=0.0)
        features_window = np.clip(features_window, -10, 10)
        label = self.labels_df.iloc[indices[0]]

        # Create mask and handle padding
        if len(features_window) < self.window_size:
            padding = np.zeros((self.window_size - len(features_window), len(self.feature_cols)))
            features_window = np.vstack([features_window, padding])
            mask = torch.zeros(self.window_size)
            mask[:len(indices)] = 1
        else:
            mask = torch.ones(self.window_size)

        # Apply augmentation if training
        if self.train and self.augmentor:
            features_window = self.augmentor.apply(features_window)
        
        return (
            torch.FloatTensor(features_window),
            torch.LongTensor([label]),
            torch.LongTensor([task_id]),
            mask
        )


class CustomLabelEncoder:
    """Custom label encoder to ensure specific mapping for health status."""

    def __init__(self):
        self.classes_ = np.array(['Sano', 'Malato'])  # 'Sano' -> 0, 'Malato' -> 1
        self.mapping_ = {'Sano': 0, 'Malato': 1}

    def fit(self, y):
        return self

    def transform(self, y):
        return np.array([self.mapping_[val.strip()] for val in y])

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)


class HandwritingDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for handwriting analysis.
    
    This class manages the complete data pipeline for handwriting analysis, including:
    - Data loading from S3
    - Preprocessing and feature engineering
    - Data splitting and cross-validation
    - Batch creation and loading
    
    The module handles both static and dynamic features, implements proper
    cross-validation splits at the subject level, and ensures balanced sampling
    for handling class imbalance.
    
    Attributes:
        s3_handler: Handler for S3 storage operations
        file_key: Key for accessing data file in S3
        config: Configuration for data processing
        column_names: Mapping of column types to their names in the dataset
        fold: Current cross-validation fold
        n_folds: Total number of cross-validation folds
        seed: Random seed for reproducibility
        feature_cols: List of feature column names
        train_dataset: Training dataset instance
        val_dataset: Validation dataset instance
        test_dataset: Test dataset instance
        encoders: Dictionary of label encoders for categorical variables
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
        """Initialize the DataModule with configuration parameters.
        
        Args:
            s3_handler: Handler for S3 operations
            file_key: Key for the data file in S3
            config: DataConfig instance containing processing parameters
            column_names: Dictionary mapping column types to their names
            fold: Current fold number (default: 0)
            n_folds: Total number of folds for cross-validation (default: 5)
            seed: Random seed for reproducibility (default: 42)
            yaml_split_path: Optional path to YAML file containing predefined splits
        """
        super().__init__()
        self.s3_handler = s3_handler
        self.file_key = file_key
        self.config = config
        self.column_names = column_names
        self.fold = fold
        self.n_folds = n_folds
        self.seed = seed
        self.yaml_split_path = yaml_split_path

        # Initialize components that will be set up later
        self.feature_cols = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Initialize encoders for categorical variables
        self.encoders = {
            'sex': LabelEncoder(),
            'work': LabelEncoder(),
            'label': CustomLabelEncoder()
        }

        if self.config.verbose:
            rprint(f"[blue]Initialized DataModule for fold {fold + 1}/{n_folds}[/blue]")

    def _load_aggregated_data(self) -> pd.DataFrame:
        """Load and validate data from S3 storage.
        
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            ValueError: If data loading fails or data is invalid
        """
        if self.config.verbose:
            rprint(f"[blue]Fetching data from S3: {self.file_key}[/blue]")
            
        try:
            data = self.s3_handler.load_data(self.file_key)
            if data is None or data.empty:
                raise ValueError("Failed to load data or data is empty")
            return data
        except Exception as e:
            rprint(f"[red]Error loading data: {str(e)}[/red]")
            raise

    def _preprocess_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess categorical variables with label encoding.
        
        This method handles the encoding of categorical variables and ensures
        consistent mapping across all data splits.
        
        Args:
            data: Input DataFrame containing categorical columns
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df = data.copy()
        columns_to_drop = []

        # Clean and encode categorical variables
        for col, suffix in [('Sex', 'Sex_encoded'), 
                          ('Work', 'Work_encoded'), 
                          ('Label', 'Label_encoded')]:
            if col in df.columns:
                df[col] = df[col].str.strip()
                encoder_key = col.lower()
                df[suffix] = self.encoders[encoder_key].fit_transform(df[col])
                columns_to_drop.append(col)
                
                if self.config.verbose:
                    if encoder_key == 'label':
                        mapping = self.encoders[encoder_key].mapping_
                    else:
                        mapping = dict(zip(
                            self.encoders[encoder_key].classes_,
                            range(len(self.encoders[encoder_key].classes_))
                        ))
                    rprint(f"[blue]Encoded {col} values: {mapping}[/blue]")

        # Remove original categorical columns
        if columns_to_drop and self.config.verbose:
            rprint(f"[yellow]Dropping original categorical columns: {columns_to_drop}[/yellow]")
        return df.drop(columns=columns_to_drop)

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training, validation, and testing.
        
        This method handles:
        1. Data loading and preprocessing
        2. Feature selection and engineering
        3. Data splitting
        4. Dataset creation
        
        Args:
            stage: Optional stage identifier ('fit' or 'test')
            
        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        try:
            if self.config.verbose:
                rprint(f"[yellow]Setting up data for fold {self.fold + 1}/{self.n_folds}...[/yellow]")

            # Load and preprocess data
            data = self._load_aggregated_data()
            data = self._preprocess_categorical(data)

            # Update column names for encoded labels
            if 'Label' in self.column_names.values():
                self.column_names = {
                    k: v + '_encoded' if v == 'Label' else v
                    for k, v in self.column_names.items()
                }

            # Validate required columns
            required_cols = [self.column_names[k] for k in ['id', 'segment', 'task', 'label']]
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Define feature columns
            self.feature_cols = [
                col for col in data.columns
                if col not in required_cols
                and col not in ['Id', 'Segment']
                and not col.endswith('_label')
            ]

            # Set up multi-index
            data.set_index([self.column_names['id'], self.column_names['segment']], 
                          inplace=True)

            # Initialize and perform stratified splitting
            splitter = StratifiedSubjectWindowKFold(
                n_splits=self.n_folds,
                test_size=0.2,  # 20% for testing
                shuffle=True,
                random_state=self.seed,
                yaml_split_path=self.yaml_split_path
            )
            
            splitter.current_fold = self.fold
            splitter.set_window_params(
                window_size=self.config.window_size,
                stride=self.config.stride,
                verbose=self.config.verbose
            )
            
            # Get splits for current fold
            splits = list(splitter.split(data, self.column_names))
            if self.fold >= len(splits):
                raise ValueError(f"Invalid fold index {self.fold}")

            train_idx, val_idx, test_idx = splits[self.fold]
            
            # Create datasets based on splits
            if stage == "fit" or stage is None:
                # Create training dataset
                train_data = data.iloc[train_idx].copy()
                self.train_dataset = HandwritingDataset(
                    data=train_data,
                    config=self.config,
                    feature_cols=self.feature_cols,
                    column_names=self.column_names,
                    normalizer=None,
                    train=True
                )

                # Create validation dataset
                val_data = data.iloc[val_idx].copy()
                self.val_dataset = HandwritingDataset(
                    data=val_data,
                    config=self.config,
                    feature_cols=self.feature_cols,
                    column_names=self.column_names,
                    normalizer=self.train_dataset.normalizer,
                    train=False
                )

            if stage == "test" or stage is None:
                # Create test dataset
                test_data = data.iloc[test_idx].copy()
                self.test_dataset = HandwritingDataset(
                    data=test_data,
                    config=self.config,
                    feature_cols=self.feature_cols,
                    column_names=self.column_names,
                    normalizer=self.train_dataset.normalizer if self.train_dataset else None,
                    train=False
                )

            if self.config.verbose:
                self._print_split_statistics(train_data, val_data, test_data)

        except Exception as e:
            rprint(f"[red]Error in setup: {str(e)}[/red]")
            raise

    def train_dataloader(self) -> DataLoader:
        """Create the training data loader with balanced batch sampling.
        
        Returns:
            DataLoader for training data
        """
        return DataLoader(
            self.train_dataset,
            batch_sampler=BalancedBatchSampler(
                self.train_dataset,
                self.config.batch_size
            ),
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation data loader.
        
        Returns:
            DataLoader for validation data
        """
        return DataLoader(
            self.val_dataset,
            batch_sampler=BalancedBatchSampler(
                self.val_dataset,
                self.config.batch_size
            ),
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False
        )

    def test_dataloader(self) -> DataLoader:
        """Create the test data loader.
        
        Returns:
            DataLoader for test data
        """
        return DataLoader(
            self.test_dataset,
            batch_sampler=BalancedBatchSampler(
                self.test_dataset,
                self.config.batch_size
            ),
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False
        )

    def get_feature_dim(self) -> int:
        """Get the dimension of the feature vector.
        
        Returns:
            Number of features in the dataset
        """
        return len(self.feature_cols)

    def _print_split_statistics(self, train_data: pd.DataFrame, 
                              val_data: pd.DataFrame,
                              test_data: pd.DataFrame) -> None:
        """Print detailed statistics about the data splits.
        
        Args:
            train_data: Training data DataFrame
            val_data: Validation data DataFrame
            test_data: Test data DataFrame
        """
        def get_stats(data):
            n_subjects = len(data.index.get_level_values(0).unique())
            n_windows = len(data)
            n_pos = (data[self.column_names['label']] == 1).sum()
            n_neg = (data[self.column_names['label']] == 0).sum()
            return n_subjects, n_windows, n_pos, n_neg

        train_stats = get_stats(train_data)
        val_stats = get_stats(val_data)
        test_stats = get_stats(test_data)

        rprint("\n[bold blue]Data Split Statistics:[/bold blue]")
        
        for name, stats in [("Training", train_stats), 
                          ("Validation", val_stats), 
                          ("Test", test_stats)]:
            rprint(f"\n{name} Set:")
            rprint(f"Subjects: {stats[0]}")
            rprint(f"Windows: {stats[1]}")
            rprint(f"Positive samples: {stats[2]}")
            rprint(f"Negative samples: {stats[3]}")
            rprint(f"Class balance: {stats[2] / (stats[2] + stats[3]):.2%} positive")
