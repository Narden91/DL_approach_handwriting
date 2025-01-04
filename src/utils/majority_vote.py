import pandas as pd
import torch
from collections import Counter


def majority_vote(predictions_df: pd.DataFrame, subject_col: str, prediction_col: str) -> pd.DataFrame:
    """
    Perform majority voting on predictions, subject-wise.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing predictions.
        subject_col (str): Column name for subjects.
        prediction_col (str): Column name for predictions.
    
    Returns:
        pd.DataFrame: DataFrame with majority vote predictions for each subject.
    """
    majority_votes = []
    
    for subject, group in predictions_df.groupby(subject_col):
        votes = group[prediction_col].tolist()
        majority_vote = Counter(votes).most_common(1)[0][0]
        majority_votes.append({
            subject_col: subject,
            prediction_col: majority_vote
        })
    
    return pd.DataFrame(majority_votes)


def get_predictions(trainer, model, dataloader):
    """Get predictions for a given dataloader using the trained model.
    
    Args:
    - trainer: PyTorch Lightning Trainer object
    - model: PyTorch Lightning Module object
    - dataloader: PyTorch DataLoader object
    
    Returns:
    - List of subjects
    - List of true labels
    - List of predicted labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_subjects = []
    with torch.no_grad():
        for batch in dataloader:
            features, labels, task_ids, masks = batch
            logits = model(features, task_ids, masks)
            preds = torch.sigmoid(logits).round().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_subjects.extend(task_ids.cpu().numpy())
    return all_subjects, all_labels, all_preds