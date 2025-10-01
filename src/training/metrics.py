import math
import numpy as np
from typing import List, Union

def accuracy_fn(
    y_true: Union[List[int], List[float], np.ndarray],
    y_pred: Union[List[int], List[float], np.ndarray],
) -> float:
    """
    Calculate accuracy given true and predicted labels.

    Args:
        y_true (Union[List[int], List[float], np.ndarray]): True labels.
        y_pred (Union[List[int], List[float], np.ndarray]): Predicted labels.
    Returns:
        float: Accuracy score.
    """
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    return correct / len(y_true) if y_true else 0.0

def precision_fn(
    y_true: Union[List[int], List[float], np.ndarray],
    y_pred: Union[List[int], List[float], np.ndarray],
) -> float:
    """
    Calculate precision given true and predicted labels.
    Args:
        y_true (Union[List[int], List[float], np.ndarray]): True labels.
        y_pred (Union[List[int], List[float], np.ndarray]): Predicted labels.
    Returns:
        float: Precision score.
    """
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def recall_fn(
    y_true: Union[List[int], List[float], np.ndarray],
    y_pred: Union[List[int], List[float], np.ndarray],
) -> float:
    """
    Calculate recall given true and predicted labels.
    Args:
        y_true (Union[List[int], List[float], np.ndarray]): True labels.
        y_pred (Union[List[int], List[float], np.ndarray]): Predicted labels.
    Returns:
        float: Recall score.
    """
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def f1_fn(
    y_true: Union[List[int], List[float], np.ndarray],
    y_pred: Union[List[int], List[float], np.ndarray],
) -> float:
    """
    Calculate F1 score given true and predicted labels.
    
    Args:
        y_true (Union[List[int], List[float], np.ndarray]): True labels.
        y_pred (Union[List[int], List[float], np.ndarray]): Predicted labels.
    Returns:
        float: F1 score.
    """
    p = precision_fn(y_true, y_pred)
    r = recall_fn(y_true, y_pred)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def roc_auc_fn(
    y_true: Union[List[int], List[float], np.ndarray],
    y_scores: Union[List[int], List[float], np.ndarray],
) -> float:
    """
    Calculate ROC AUC given true labels and predicted scores.
    
    Args:
        y_true (Union[List[int], List[float], np.ndarray]): True binary labels (0 or 1).
        y_scores (Union[List[int], List[float], np.ndarray]): Predicted scores (probabilities or confidence).
    Returns:
        float: ROC AUC score.
    """
    desc = sorted(zip(y_scores, y_true), key=lambda x: (-x[0], -x[1]))
    pos = sum(y_true)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.0
    rank = 1
    sum_ranks = 0
    for score, label in desc:
        if label == 1:
            sum_ranks += rank
        rank += 1
    auc = (sum_ranks - pos * (pos + 1) / 2) / (pos * neg)
    return auc
