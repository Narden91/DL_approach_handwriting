from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
import torch
from torch.utils.data import Dataset, DataLoader
from rich import print as rprint
from s3_operations.s3_io import S3IOHandler
from src.data.balanced_batch import BalancedBatchSampler
from src.data.stratified_k_fold import StratifiedSubjectWindowKFold


class HandwritingDataset(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            window_size: int,
            stride: int,
            feature_cols: List[str],
            column_names: Dict[str, str],
            scaler: Optional[object] = None,
            scaler_type: str = "standard",
            train: bool = True,
            verbose: bool = True
    ):
        """Initialize the HandwritingDataset."""
        self.data = data.copy()
        self.window_size = window_size
        self.stride = stride
        self.feature_cols = feature_cols
        self.column_names = column_names
        self.train = train

        rprint(f"Creating dataset:\n {self.data}") if verbose else None

        # Separate features and labels
        self.features_df = self.data[feature_cols]
        self.labels_df = self.data[self.column_names['label']]

        # Initialize and apply normalization
        if scaler is None and train:
            self.scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
            self.features_df = pd.DataFrame(
                self.scaler.fit_transform(self.features_df),
                columns=self.features_df.columns,
                index=self.features_df.index
            )
            rprint(f"Fitted new {self.scaler.__class__.__name__} on training data") if verbose else None
        elif scaler is not None:
            self.scaler = scaler
            self.features_df = pd.DataFrame(
                self.scaler.transform(self.features_df),
                columns=self.features_df.columns,
                index=self.features_df.index
            )

        # Create windows for each subject and task
        self.windows = self._create_windows()
        rprint(f"Created dataset with {len(self.windows)} windows") if verbose else None

    def _normalize_features(self):
        """Additional normalization step"""
        feature_means = self.features_df.mean()
        feature_stds = self.features_df.std()
        self.features_df = (self.features_df - feature_means) / (feature_stds + 1e-8)
        return self.features_df

    def _create_windows(self) -> List[Tuple[int, int, int, List[int]]]:
        """Create sliding windows for each subject and task combination."""
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
        subject_id, task_id, window_size, indices = self.windows[idx]

        # Handle NaN values and extreme values
        features_window = np.nan_to_num(self.features_df.iloc[indices].values, nan=0.0)
        features_window = np.clip(features_window, -10, 10)
        label = self.labels_df.iloc[indices[0]]

        # Padding if necessary
        if len(features_window) < self.window_size:
            padding = np.zeros((self.window_size - len(features_window), len(self.feature_cols)))
            features_window = np.vstack([features_window, padding])
            mask = torch.zeros(self.window_size)
            mask[:len(indices)] = 1
        else:
            mask = torch.ones(self.window_size)

        # Window normalization
        window_mean = np.nanmean(features_window, axis=0)
        window_std = np.nanstd(features_window, axis=0) + 1e-8
        features_window = (features_window - window_mean) / window_std

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
    def __init__(
            self,
            s3_handler: S3IOHandler,
            file_key: str,
            batch_size: int,
            window_size: int,
            stride: int,
            num_workers: int,
            num_tasks: int,
            val_size: float,
            test_size: float,
            column_names: Dict[str, str],
            fold: int = 0,
            n_folds: int = 5,
            scaler_type: str = "standard",
            seed: int = 42,
            verbose: bool = True
    ):
        super().__init__()

        self.s3_handler = s3_handler
        self.file_key = file_key
        self.batch_size = batch_size
        self.window_size = window_size
        self.stride = stride
        self.num_workers = num_workers
        self.num_tasks = num_tasks
        self.val_size = val_size
        self.test_size = test_size
        self.column_names = column_names
        self.fold = fold
        self.n_folds = n_folds
        self.scaler_type = scaler_type.lower()
        self.seed = seed
        self.verbose = verbose

        self.scaler = None
        self.feature_cols = None
        self.encoders = {
            'sex': LabelEncoder(),
            'work': LabelEncoder(),
            'label': CustomLabelEncoder()
        }

        self.train_dataset = None
        self.val_dataset = None
        self.class_weights = None
        self.test_dataset = None

        if self.verbose:
            rprint(f"[blue]Initialized DataModule for fold {fold + 1}/{n_folds}[/blue]")

    def _load_aggregated_data(self) -> pd.DataFrame:
        """Load aggregated data using S3IOHandler."""
        rprint(
            f"[blue]Fetching aggregated data from S3: {self.file_key} in Bucket: {self.s3_handler.bucket_name}[/blue]") if self.verbose else None
        return self.s3_handler.load_data(self.file_key)

    def _preprocess_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess categorical columns by removing whitespace and encoding.
        After encoding, removes original categorical columns.
        """
        df = data.copy()

        # Strip whitespace from all string columns
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].str.strip()

        # List to keep track of columns to drop
        columns_to_drop = []

        # Encode categorical variables
        if 'Sex' in df.columns:
            df['Sex_encoded'] = self.encoders['sex'].fit_transform(df['Sex'])
            columns_to_drop.append('Sex')
            rprint(
                f"[blue]Encoded Sex values: {dict(zip(self.encoders['sex'].classes_, range(len(self.encoders['sex'].classes_))))}[/blue]") if self.verbose else None

        if 'Work' in df.columns:
            df['Work_encoded'] = self.encoders['work'].fit_transform(df['Work'])
            columns_to_drop.append('Work')
            rprint(
                f"[blue]Encoded Work values: {dict(zip(self.encoders['work'].classes_, range(len(self.encoders['work'].classes_))))}[/blue]") if self.verbose else None

        if 'Label' in df.columns:
            df['Label_encoded'] = self.encoders['label'].fit_transform(df['Label'])
            columns_to_drop.append('Label')
            rprint(f"[blue]Encoded Label values: {self.encoders['label'].mapping_}[/blue]") if self.verbose else None

        # Drop original categorical columns
        if columns_to_drop:
            rprint(
                f"[yellow]Dropping original categorical columns: {columns_to_drop}[/yellow]") if self.verbose else None
            df = df.drop(columns=columns_to_drop)

        return df

    def setup(self, stage: Optional[str] = None):
        """Load and preprocess the data with stratified subject distribution."""
        try:
            if self.verbose:
                rprint(f"[yellow]Setting up data for fold {self.fold + 1}/{self.n_folds}...[/yellow]")

            # Load and preprocess data
            data = self._load_aggregated_data()
            if data is None or data.empty:
                raise ValueError("Failed to load data or data is empty")

            data = self._preprocess_categorical(data)

            # Update column names for encoded labels
            if 'Label' in self.column_names.values():
                self.column_names = {k: v + '_encoded' if v == 'Label' else v
                                     for k, v in self.column_names.items()}

            # Validate required columns exist
            required_cols = [self.column_names[k] for k in ['id', 'segment', 'task', 'label']]
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Define metadata and feature columns
            metadata_cols = required_cols
            self.feature_cols = [col for col in data.columns
                                 if col not in metadata_cols
                                 and col not in ['Id', 'Segment']
                                 and not col.endswith('_label')]

            if not self.feature_cols:
                raise ValueError("No feature columns identified")

            # Use index columns but don't verify integrity to allow duplicates
            data.set_index([self.column_names['id'], self.column_names['segment']], inplace=True)

            # Validate subjects count
            unique_subjects = data.index.get_level_values(0).unique()
            total_subjects = len(unique_subjects)
            if total_subjects != 174:
                raise ValueError(f"Expected 174 subjects, got {total_subjects}")

            # Initialize the stratified k-fold splitter
            splitter = StratifiedSubjectWindowKFold(
                n_splits=self.n_folds,
                test_size=self.test_size,
                shuffle=True,
                random_state=self.seed
            )

            # Set window parameters
            splitter.set_window_params(
                window_size=self.window_size,
                stride=self.stride,
                verbose=self.verbose
            )

            # Get splits for current fold
            splits = list(splitter.split(data, self.column_names))

            if self.fold >= len(splits):
                raise ValueError(f"Invalid fold index {self.fold}. Only {len(splits)} folds available.")

            train_idx, val_idx, test_idx = splits[self.fold]

            # Verify non-empty splits
            if len(train_idx) == 0:
                raise ValueError(f"Empty training split detected for fold {self.fold + 1}")
            if len(val_idx) == 0:
                raise ValueError(f"Empty validation split detected for fold {self.fold + 1}")
            if len(test_idx) == 0:
                raise ValueError(f"Empty test split detected for fold {self.fold + 1}")

            # Split data
            train_data = data.iloc[train_idx].copy()
            val_data = data.iloc[val_idx].copy()
            test_data = data.iloc[test_idx].copy()

            if self.verbose:
                rprint(f"\n[blue]Data split sizes:[/blue]")
                rprint(
                    f"Training data: {len(train_data)} segments, {len(train_data.index.get_level_values(0).unique())} subjects")
                rprint(
                    f"Validation data: {len(val_data)} segments, {len(val_data.index.get_level_values(0).unique())} subjects")
                rprint(
                    f"Test data: {len(test_data)} segments, {len(test_data.index.get_level_values(0).unique())} subjects")

            # Initialize training dataset first
            if stage == "fit" or stage is None:
                self.train_dataset = self._create_dataset(
                    data=train_data,
                    is_train=True,
                    scaler=None
                )

                # Initialize validation dataset
                self.val_dataset = self._create_dataset(
                    data=val_data,
                    is_train=False,
                    scaler=self.train_dataset.scaler
                )

                # Compute and store class weights
                labels = train_data[self.column_names['label']]
                class_counts = labels.value_counts()
                self.class_weights = {
                    label: len(labels) / (2 * count)
                    for label, count in class_counts.items()
                }

                if self.verbose:
                    rprint("\n[blue]Class weights:[/blue]")
                    for label, weight in self.class_weights.items():
                        rprint(f"Class {label}: {weight:.4f}")

            # Initialize test dataset if needed
            if stage == "test" or stage is None:
                self.test_dataset = self._create_dataset(
                    data=test_data,
                    is_train=False,
                    scaler=self.train_dataset.scaler if hasattr(self, 'train_dataset') else None
                )

        except Exception as e:
            rprint(f"[red]Error in setup: {str(e)}[/red]")
            raise


    def _create_dataset(self, data: pd.DataFrame, is_train: bool,
                        scaler: Optional[object] = None) -> HandwritingDataset:
        """Helper method to create and validate datasets."""
        if data.empty:
            raise ValueError(f"Empty data for {'training' if is_train else 'validation/testing'} dataset")

        return HandwritingDataset(
            data=data,
            window_size=self.window_size,
            stride=self.stride,
            feature_cols=self.feature_cols,
            column_names=self.column_names,
            scaler=scaler,
            scaler_type=self.scaler_type,
            train=is_train,
            verbose=self.verbose
        )

    def set_class_weights(self, dataset):
        if hasattr(dataset, 'class_weights'):
            self.class_weights = torch.tensor([
                dataset.class_weights[0],
                dataset.class_weights[1]
            ], device=self.device)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_sampler=BalancedBatchSampler(
                self.train_dataset,
                self.batch_size
            ),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_sampler=BalancedBatchSampler(
                self.val_dataset,
                self.batch_size
            ),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_sampler=BalancedBatchSampler(
                self.test_dataset,
                self.batch_size
            ),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def get_feature_dim(self) -> int:
        """Get the dimension of the feature vector."""
        return len(self.feature_cols)

    def get_num_tasks(self) -> int:
        """Get the number of unique tasks."""
        return self.num_tasks

    def _print_split_statistics(self, train_data, val_data, test_data=None):
        """Print detailed statistics about the data splits"""

        def get_stats(data):
            n_subjects = len(data.index.get_level_values(0).unique())
            n_windows = len(data)
            n_pos = (data[self.column_names['label']] == 1).sum()
            n_neg = (data[self.column_names['label']] == 0).sum()
            return n_subjects, n_windows, n_pos, n_neg

        train_stats = get_stats(train_data)
        val_stats = get_stats(val_data)

        rprint("\n[bold blue]Data Split Statistics:[/bold blue]")
        rprint(f"\nTraining Set:")
        rprint(f"Subjects: {train_stats[0]}")
        rprint(f"Windows: {train_stats[1]}")
        rprint(f"Positive samples: {train_stats[2]}")
        rprint(f"Negative samples: {train_stats[3]}")

        rprint(f"\nValidation Set:")
        rprint(f"Subjects: {val_stats[0]}")
        rprint(f"Windows: {val_stats[1]}")
        rprint(f"Positive samples: {val_stats[2]}")
        rprint(f"Negative samples: {val_stats[3]}")

        if test_data is not None:
            test_stats = get_stats(test_data)
            rprint(f"\nTest Set:")
            rprint(f"Subjects: {test_stats[0]}")
            rprint(f"Windows: {test_stats[1]}")
            rprint(f"Positive samples: {test_stats[2]}")
            rprint(f"Negative samples: {test_stats[3]}")
