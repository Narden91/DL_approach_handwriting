import pandas as pd
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