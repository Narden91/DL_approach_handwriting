"""
This is the demo code that uses hydra to access the parameters in under the directory config.
"""
import hydra
from omegaconf import DictConfig
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from utils import HandwritingLSTM, train_model, evaluate_model, print_fold_metrics

@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def train_and_evaluate(cfg: DictConfig):
    """Function to train and evaluate the model"""
    processed_dir = Path(cfg.data.processed_dir).resolve()
    print(f"Training model using data from {processed_dir}")
    print(f"Model configuration: {cfg.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load processed data
    train_data = pd.read_csv(processed_dir / 'train.csv')
    val_data = pd.read_csv(processed_dir / 'val.csv')
    test_data = pd.read_csv(processed_dir / 'test.csv')

    X_train, y_train = train_data.drop(cfg.process.label_column, axis=1), train_data[cfg.process.label_column]
    X_val, y_val = val_data.drop(cfg.process.label_column, axis=1), val_data[cfg.process.label_column]
    X_test, y_test = test_data.drop(cfg.process.label_column, axis=1), test_data[cfg.process.label_column]

    # Create data loaders
    train_loader = DataLoader(TensorDataset(torch.Tensor(X_train.values), torch.LongTensor(y_train.values)), batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.Tensor(X_val.values), torch.LongTensor(y_val.values)), batch_size=cfg.training.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.Tensor(X_test.values), torch.LongTensor(y_test.values)), batch_size=cfg.training.batch_size, shuffle=False)

    # Initialize model
    model = HandwritingLSTM(
        cfg.model.input_size, 
        cfg.model.hidden_size, 
        cfg.model.num_layers, 
        cfg.model.num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, cfg.training.num_epochs, device)

    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")

    # Save the model
    final_dir = Path(cfg.data.final_dir).resolve()
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_dir / 'handwriting_model.pth')
    print(f"Model saved to {final_dir / 'handwriting_model.pth'}")

    # Perform cross-validation
    print("\nPerforming cross-validation:")
    skf = StratifiedKFold(n_splits=cfg.training.num_folds, shuffle=True, random_state=42)
    cv_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"Fold {fold}")

        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        train_loader = DataLoader(TensorDataset(torch.Tensor(X_train_fold.values), torch.LongTensor(y_train_fold.values)), batch_size=cfg.training.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.Tensor(X_val_fold.values), torch.LongTensor(y_val_fold.values)), batch_size=cfg.training.batch_size, shuffle=False)

        model = HandwritingLSTM(cfg.model.input_size, cfg.model.hidden_size, cfg.model.num_layers, cfg.model.num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

        train_model(model, train_loader, val_loader, criterion, optimizer, cfg.training.num_epochs, device)

        accuracy, precision, recall, f1 = evaluate_model(model, val_loader, device)
        cv_scores['accuracy'].append(accuracy)
        cv_scores['precision'].append(precision)
        cv_scores['recall'].append(recall)
        cv_scores['f1'].append(f1)

    print("\nCross-validation results:")
    print_fold_metrics(cv_scores)

if __name__ == "__main__":
    train_and_evaluate()