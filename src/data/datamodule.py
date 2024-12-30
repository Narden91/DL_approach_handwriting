from pathlib import Path
from typing import List, Optional, Tuple, Dict
import boto3
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
import torch
from torch.utils.data import Dataset, DataLoader
from rich import print as rprint

from s3_operations.s3_io import S3IOHandler
# from s3_operations.s3_handler import config


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
        
        if self.verbose:
            rprint(f"[blue]Initialized DataModule for fold {fold + 1}/{n_folds}[/blue]")
            
    def _load_aggregated_data(self) -> pd.DataFrame:
        """Load aggregated data using S3IOHandler."""
        rprint(f"[blue]Fetching aggregated data from S3: {self.file_key} in Bucket: {self.s3_handler.bucket_name}[/blue]") if self.verbose else None
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
            rprint(f"[blue]Encoded Sex values: {dict(zip(self.encoders['sex'].classes_, range(len(self.encoders['sex'].classes_))))}[/blue]") if self.verbose else None
        
        if 'Work' in df.columns:
            df['Work_encoded'] = self.encoders['work'].fit_transform(df['Work'])
            columns_to_drop.append('Work')
            rprint(f"[blue]Encoded Work values: {dict(zip(self.encoders['work'].classes_, range(len(self.encoders['work'].classes_))))}[/blue]") if self.verbose else None
        
        if 'Label' in df.columns:
            df['Label_encoded'] = self.encoders['label'].fit_transform(df['Label'])
            columns_to_drop.append('Label')
            rprint(f"[blue]Encoded Label values: {self.encoders['label'].mapping_}[/blue]") if self.verbose else None
        
        # Drop original categorical columns
        if columns_to_drop:
            rprint(f"[yellow]Dropping original categorical columns: {columns_to_drop}[/yellow]") if self.verbose else None
            df = df.drop(columns=columns_to_drop)
        
        return df

    def setup(self, stage: Optional[str] = None):
        """Load and preprocess the data with correct subject distribution."""
        rprint(f"[yellow]Setting up data for fold {self.fold + 1}/{self.n_folds}...[/yellow]") if self.verbose else None
        
        # Load and preprocess data
        data = self._load_aggregated_data()
        data = self._preprocess_categorical(data)
        
        rprint(f"[blue]Data loaded and preprocessed successfully:[/blue]\n{data}") if self.verbose else None
        
        # Update column names for encoded labels
        if 'Label' in self.column_names.values():
            self.column_names = {k: v + '_encoded' if v == 'Label' else v 
                            for k, v in self.column_names.items()}
        
        # Define metadata columns
        metadata_cols = [
            self.column_names['id'],
            self.column_names['segment'],
            self.column_names['task'],
            self.column_names['label']
        ]
        
        # Set feature columns
        self.feature_cols = [col for col in data.columns 
                            if col not in metadata_cols 
                            and col not in ['Id', 'Segment']
                            and not col.endswith('_label')]
        
        # Set multi-index
        data = data.set_index([self.column_names['id'], self.column_names['segment']])
        
        # Get unique subjects and verify count
        unique_subjects = data.index.get_level_values(0).unique()
        total_subjects = len(unique_subjects)
        assert total_subjects == 174, f"Expected 174 subjects, got {total_subjects}"
        
        # Calculate split sizes (70/15/15)
        test_size = int(self.test_size * total_subjects) 
        val_size = int(self.val_size * total_subjects)  
        train_size = total_subjects - (test_size + val_size)
        
        # Create splits
        np.random.seed(self.seed)
        shuffled_subjects = np.random.permutation(unique_subjects)
        
        if stage == "fit" or stage is None:
            # Create train/val split
            train_subjects = shuffled_subjects[:train_size]
            val_subjects = shuffled_subjects[train_size:train_size + val_size]
            
            train_data = data[data.index.get_level_values(0).isin(train_subjects)]
            val_data = data[data.index.get_level_values(0).isin(val_subjects)]
            
            self.train_dataset = HandwritingDataset(
                data=train_data,
                window_size=self.window_size,
                stride=self.stride,
                feature_cols=self.feature_cols,
                column_names=self.column_names,
                scaler_type=self.scaler_type,
                train=True,
                verbose=self.verbose
            )
            
            self.val_dataset = HandwritingDataset(
                data=val_data,
                window_size=self.window_size,
                stride=self.stride,
                feature_cols=self.feature_cols,
                column_names=self.column_names,
                scaler=self.train_dataset.scaler,
                scaler_type=self.scaler_type,
                train=False,
                verbose=self.verbose
            )

        if stage == "test" or stage is None:
            # Create test split
            test_subjects = shuffled_subjects[train_size + val_size:]
            test_data = data[data.index.get_level_values(0).isin(test_subjects)]
            
            self.test_dataset = HandwritingDataset(
                data=test_data,
                window_size=self.window_size,
                stride=self.stride,
                feature_cols=self.feature_cols,
                column_names=self.column_names,
                scaler=self.train_dataset.scaler if hasattr(self, 'train_dataset') else None,
                scaler_type=self.scaler_type,
                train=False,
                verbose=self.verbose
            )
        
        # Verify splits
        if stage is None:
            train_ids = set(train_subjects)
            val_ids = set(val_subjects)
            test_ids = set(test_subjects)
            
            assert len(train_ids & val_ids) == 0, "Overlap found between train and val sets"
            assert len(train_ids & test_ids) == 0, "Overlap found between train and test sets"
            assert len(val_ids & test_ids) == 0, "Overlap found between val and test sets"
            
            if self.verbose:
                rprint(f"[green]Dataset split successful:[/green]")
                rprint(f"Train subjects: {len(train_ids)}")
                rprint(f"Val subjects: {len(val_ids)}")
                rprint(f"Test subjects: {len(test_ids)}")
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    
    def get_feature_dim(self) -> int:
        """Get the dimension of the feature vector."""
        return len(self.feature_cols)
    
    def get_num_tasks(self) -> int:
        """Get the number of unique tasks."""
        return self.num_tasks