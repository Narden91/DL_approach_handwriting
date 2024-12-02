from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from rich import print as rprint


class HandwritingDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int,
        stride: int,
        feature_cols: List[str],
        column_names: Dict[str, str],
        scaler: Optional[StandardScaler] = None,
        train: bool = True
    ):
        """Initialize the HandwritingDataset."""
        # Make a deep copy of the data to avoid SettingWithCopyWarning
        self.data = data.copy()
        self.window_size = window_size
        self.stride = stride
        self.feature_cols = feature_cols
        self.column_names = column_names
        self.train = train
        
        # Normalize features
        if scaler is None and train:
            self.scaler = StandardScaler()
            self.data.loc[:, feature_cols] = self.scaler.fit_transform(self.data[feature_cols])
            rprint("[green]Fitted new StandardScaler on training data[/green]")
        elif scaler is not None:
            self.scaler = scaler
            self.data.loc[:, feature_cols] = self.scaler.transform(self.data[feature_cols])
            rprint("[green]Applied existing StandardScaler to data[/green]")
        
        # Create windows for each subject and task
        self.windows = self._create_windows()
        
        rprint(f"[blue]Created dataset with {len(self.windows)} windows[/blue]")
    
    def _create_windows(self) -> List[Tuple[int, int, int, int]]:
        """Create sliding windows for each subject and task combination."""
        windows = []
        id_col = self.column_names['id']
        task_col = self.column_names['task']
        
        for subject in self.data[id_col].unique():
            for task in self.data[task_col].unique():
                subject_task_data = self.data[
                    (self.data[id_col] == subject) & 
                    (self.data[task_col] == task)
                ]
                
                if len(subject_task_data) == 0:
                    continue
                
                if len(subject_task_data) < self.window_size:
                    windows.append((subject, task, 0, len(subject_task_data)))
                else:
                    for start in range(0, len(subject_task_data) - self.window_size + 1, self.stride):
                        end = start + self.window_size
                        windows.append((subject, task, start, end))
        
        return windows
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        subject_id, task_id, start_idx, end_idx = self.windows[idx]
        id_col = self.column_names['id']
        task_col = self.column_names['task']
        label_col = self.column_names['label']
        
        # Get data for this window
        window_data = self.data[
            (self.data[id_col] == subject_id) & 
            (self.data[task_col] == task_id)
        ].iloc[start_idx:end_idx]
        
        # Extract features and label
        features = window_data[self.feature_cols].values
        label = window_data[label_col].iloc[0]
        
        # Pad if necessary
        if len(features) < self.window_size:
            padding = np.zeros((self.window_size - len(features), len(self.feature_cols)))
            features = np.vstack([features, padding])
            
            # Create attention mask (1 for real data, 0 for padding)
            mask = torch.zeros(self.window_size)
            mask[:len(window_data)] = 1
        else:
            mask = torch.ones(self.window_size)
        
        # Convert to tensors
        features = torch.FloatTensor(features)
        label = torch.LongTensor([label])
        task = torch.LongTensor([task_id])
        
        return features, label, task, mask


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
        data_dir: str,
        batch_size: int,
        window_size: int,
        stride: int,
        num_workers: int,
        val_split: float,
        test_split: float,
        num_tasks: int,
        file_pattern: str,
        column_names: Dict[str, str],
        seed: int = 42
    ):
        """Initialize the data module."""
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.window_size = window_size
        self.stride = stride
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.num_tasks = num_tasks
        self.file_pattern = file_pattern
        self.column_names = column_names
        self.seed = seed
        
        self.scaler = None
        self.feature_cols = None
        self.encoders = {
            'sex': LabelEncoder(),
            'work': LabelEncoder(),
            'label': CustomLabelEncoder()  # Using custom encoder for Label
        }

    def _preprocess_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess categorical columns by removing whitespace and encoding."""
        df = df.copy()
        
        # Strip whitespace from all string columns
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].str.strip()
        
        # Encode categorical variables
        if 'Sex' in df.columns:
            df['Sex_encoded'] = self.encoders['sex'].fit_transform(df['Sex'])
            rprint(f"[blue]Encoded Sex values: {dict(zip(self.encoders['sex'].classes_, range(len(self.encoders['sex'].classes_))))}[/blue]")
        
        if 'Work' in df.columns:
            df['Work_encoded'] = self.encoders['work'].fit_transform(df['Work'])
            rprint(f"[blue]Encoded Work values: {dict(zip(self.encoders['work'].classes_, range(len(self.encoders['work'].classes_))))}[/blue]")
        
        if 'Label' in df.columns:
            df['Label_encoded'] = self.encoders['label'].fit_transform(df['Label'])
            rprint(f"[blue]Encoded Label values: {self.encoders['label'].mapping_}[/blue]")
        
        return df
    
    def _get_aggregated_data_path(self) -> Path:
        """Get the path for the aggregated raw data CSV file."""
        return Path(self.data_dir) / "aggregated_data.csv"

    def _load_or_aggregate_data(self) -> pd.DataFrame:
        """Load aggregated data if exists, otherwise aggregate from raw files."""
        aggregated_path = self._get_aggregated_data_path()

        # Check if aggregated data exists
        if aggregated_path.exists():
            rprint("[green]Loading existing aggregated data...[/green]")
            return pd.read_csv(aggregated_path)
        
        rprint("[yellow]Aggregating data from raw files...[/yellow]")
        # Load and aggregate all CSV files
        dfs = []
        for i in range(1, self.num_tasks + 1):
            file_path = self.data_dir / self.file_pattern.format(i)
            rprint(f"[blue]Loading {file_path}[/blue]")
            df = pd.read_csv(file_path)
            
            if self.column_names['task'] not in df.columns:
                df[self.column_names['task']] = i
            dfs.append(df)
        
        # Concatenate all dataframes
        data = pd.concat(dfs, ignore_index=True)
        
        # Save aggregated data
        data.to_csv(aggregated_path, index=False)
        rprint(f"[green]Saved aggregated data to {aggregated_path}[/green]")
        
        return data

    def setup(self, stage: Optional[str] = None):
        """Load and preprocess the data."""
        rprint("[yellow]Setting up data...[/yellow]")
        
        # Load or aggregate data
        data = self._load_or_aggregate_data()
        
        # Preprocess categorical variables
        data = self._preprocess_categorical(data)
        
        # Update column names to use encoded versions
        if 'Label' in self.column_names.values():
            self.column_names = {k: v + '_encoded' if v == 'Label' else v 
                               for k, v in self.column_names.items()}
        
        # Identify feature columns
        exclude_cols = (
            list(self.column_names.values()) + 
            ['Sex', 'Work', 'Label'] +  # Original categorical columns
            ['Sex_encoded', 'Work_encoded', 'Label_encoded']  # Encoded columns
        )
        self.feature_cols = [col for col in data.columns if col not in exclude_cols]
        self.feature_cols.extend(['Sex_encoded', 'Work_encoded'])
        
        # Split data into train, val, test
        unique_subjects = data[self.column_names['id']].unique()
        np.random.seed(self.seed)
        np.random.shuffle(unique_subjects)
        
        n_subjects = len(unique_subjects)
        n_test = int(n_subjects * self.test_split)
        n_val = int(n_subjects * self.val_split)
        
        test_subjects = unique_subjects[:n_test]
        val_subjects = unique_subjects[n_test:n_test + n_val]
        train_subjects = unique_subjects[n_test + n_val:]
        
        # Create datasets
        if stage == 'fit' or stage is None:
            train_data = data[data[self.column_names['id']].isin(train_subjects)]
            self.train_dataset = HandwritingDataset(
                data=train_data,
                window_size=self.window_size,
                stride=self.stride,
                feature_cols=self.feature_cols,
                column_names=self.column_names,
                train=True
            )
            self.scaler = self.train_dataset.scaler
            
            val_data = data[data[self.column_names['id']].isin(val_subjects)]
            self.val_dataset = HandwritingDataset(
                data=val_data,
                window_size=self.window_size,
                stride=self.stride,
                feature_cols=self.feature_cols,
                column_names=self.column_names,
                scaler=self.scaler,
                train=False
            )
        
        if stage == 'test' or stage is None:
            test_data = data[data[self.column_names['id']].isin(test_subjects)]
            self.test_dataset = HandwritingDataset(
                data=test_data,
                window_size=self.window_size,
                stride=self.stride,
                feature_cols=self.feature_cols,
                column_names=self.column_names,
                scaler=self.scaler,
                train=False
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_feature_dim(self) -> int:
        """Get the dimension of the feature vector."""
        return len(self.feature_cols)
    
    def get_num_tasks(self) -> int:
        """Get the number of unique tasks."""
        return self.num_tasks