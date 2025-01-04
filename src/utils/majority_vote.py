import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
import torch
from collections import Counter
from rich import print as rprint


def get_predictions(trainer, model, dataloader):
    """Get predictions for all windows in a dataloader."""
    device = next(model.parameters()).device
    all_preds = []
    all_labels = []
    all_subjects = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            features, labels, task_ids, masks = batch
            features = features.to(device)
            task_ids = task_ids.to(device)
            masks = masks.to(device)
            
            logits = model(features, task_ids, masks)
            preds = torch.sigmoid(logits) > 0.5
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_subjects.extend(task_ids.cpu().numpy().flatten())
            
    return np.array(all_subjects), np.array(all_labels), np.array(all_preds)


def aggregate_subject_predictions(subjects, predictions):
    """
    Aggregate window predictions to subject level using majority vote.
    Returns both the binary predictions and the raw probabilities.
    """
    subject_preds = {}
    for subject, pred in zip(subjects.astype(int), predictions):
        if subject not in subject_preds:
            subject_preds[subject] = []
        subject_preds[subject].append(float(pred))
    
    # Calculate both mean probabilities and binary predictions
    subject_probs = {subject: np.mean(preds) for subject, preds in subject_preds.items()}
    subject_binary = {subject: prob > 0.5 for subject, prob in subject_probs.items()}
    
    return subject_binary, subject_probs


def calculate_confusion_matrix(y_true, y_pred, verbose=True):
    """Calculate confusion matrix and return all metrics with validation."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})")
        
    # Convert to numpy arrays and ensure binary values
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    
    # Calculate confusion matrix elements
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Print confusion matrix for debugging
    if verbose:
        rprint(f"\n[blue]Confusion Matrix:[/blue]")
        rprint(f"True Positives: {tp}")
        rprint(f"True Negatives: {tn}")
        rprint(f"False Positives: {fp}")
        rprint(f"False Negatives: {fn}")
    
    # Calculate metrics with safe division
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate MCC with safe handling of zero divisions
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'mcc': mcc,
        'confusion_matrix': {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
    }


def compute_subject_metrics(true_subjects, true_labels, predictions, verbose=True):
    """Compute comprehensive metrics at subject level with validation checks."""
    try:
        # Get binary predictions and probabilities
        subject_preds, subject_probs = aggregate_subject_predictions(true_subjects, predictions)
        subject_true, _ = aggregate_subject_predictions(true_subjects, true_labels)
        
        # Convert to arrays for metrics calculation
        subjects = sorted(subject_true.keys())
        y_true = np.array([subject_true[s] for s in subjects])
        y_pred = np.array([subject_preds[s] for s in subjects])
        
        # Print distribution information
        if verbose:
            rprint(f"\n[green]Label Distribution:[/green]")
            rprint(f"True labels distribution: {np.bincount(y_true.astype(int))}")
            rprint(f"Predicted labels distribution: {np.bincount(y_pred.astype(int))}")
        
        # Calculate all metrics using confusion matrix
        metrics = calculate_confusion_matrix(y_true, y_pred, verbose=verbose)
        
        return {
            'subject_accuracy': metrics['accuracy'],
            'subject_precision': metrics['precision'],
            'subject_recall': metrics['recall'],
            'subject_specificity': metrics['specificity'],
            'subject_f1': metrics['f1'],
            'subject_mcc': metrics['mcc']
        }
        
    except Exception as e:
        rprint(f"[red]Error in compute_subject_metrics: {str(e)}[/red]")
        raise