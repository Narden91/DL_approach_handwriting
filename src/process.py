"""
This file contains the code for processing data.

The `process_data` function loads raw data from multiple CSV files, preprocesses it, splits it into train, validation, and test sets,
and saves the processed data into separate CSV files.

"""
import hydra
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def process_data(cfg: DictConfig):
    """Function to process the data"""
    raw_dir = Path(cfg.data.raw_dir).resolve()
    print(f"Process data using {raw_dir}")
    print(f"Columns used: {cfg.process.use_columns}")

    # Load data
    all_data = []
    for i in range(1, 35):  # T01 to T34
        file_path = raw_dir / f'T{i:02d}.csv'
        if file_path.is_file():
            data = pd.read_csv(file_path)
            data['Task'] = i  # Add a column to identify the task
            all_data.append(data)
        else:
            print(f"Warning: {file_path} not found.")
    
    if not all_data:
        raise FileNotFoundError(f"No data files found in {raw_dir}. Please check the path and ensure the CSV files are present.")

    raw_data = pd.concat(all_data, ignore_index=True)

    # Preprocess data
    numerical_columns = cfg.process.use_columns
    scaler = StandardScaler()
    raw_data[numerical_columns] = scaler.fit_transform(raw_data[numerical_columns])

    categorical_columns = cfg.process.categorical_columns
    le = LabelEncoder()
    for col in categorical_columns:
        raw_data[col] = le.fit_transform(raw_data[col])

    raw_data[cfg.process.label_column] = le.fit_transform(raw_data[cfg.process.label_column])

    # Split data
    X = raw_data.drop(cfg.process.label_column, axis=1)
    y = raw_data[cfg.process.label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.process.test_ratio, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=cfg.process.val_ratio/(1-cfg.process.test_ratio), stratify=y_train, random_state=42)

    # Save processed data
    processed_dir = Path(cfg.data.processed_dir).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving processed data to: {processed_dir}")

    pd.concat([X_train, y_train], axis=1).to_csv(processed_dir / 'train.csv', index=False)
    pd.concat([X_val, y_val], axis=1).to_csv(processed_dir / 'val.csv', index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(processed_dir / 'test.csv', index=False)

    print("Data processing completed.")

if __name__ == "__main__":
    process_data()