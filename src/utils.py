import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabulate import tabulate
import numpy as np

class HandwritingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(HandwritingLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return accuracy, precision, recall, f1

def print_fold_metrics(cv_scores):
    headers = ["Fold", "Accuracy", "Precision", "Recall", "F1-Score"]
    table_data = []
    
    for fold, (accuracy, precision, recall, f1) in enumerate(zip(
        cv_scores['accuracy'], cv_scores['precision'], cv_scores['recall'], cv_scores['f1']
    ), 1):
        table_data.append([
            fold,
            f"{accuracy:.4f}",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1:.4f}"
        ])
    
    # Add average row
    avg_row = [
        "Average",
        f"{np.mean(cv_scores['accuracy']):.4f} ± {np.std(cv_scores['accuracy']):.4f}",
        f"{np.mean(cv_scores['precision']):.4f} ± {np.std(cv_scores['precision']):.4f}",
        f"{np.mean(cv_scores['recall']):.4f} ± {np.std(cv_scores['recall']):.4f}",
        f"{np.mean(cv_scores['f1']):.4f} ± {np.std(cv_scores['f1']):.4f}"
    ]
    table_data.append(avg_row)

    print(tabulate(table_data, headers=headers, tablefmt="grid"))