"""Prediction module for evaluating model outputs."""

from typing import Dict, Union

import numpy as np
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(
    y_true: Union[np.ndarray, torch.Tensor, list],
    y_pred: Union[np.ndarray, torch.Tensor, list],
    y_prob: Union[np.ndarray, torch.Tensor, list],
) -> Dict[str, float]:
    """
    Calculate a variety of performance metrics given true labels, predicted labels,
    and predicted probabilities.

    This function computes several commonly used classification metrics to evaluate
    the performance of a model. It returns a dictionary containing the balanced
    accuracy, F1 score, precision, recall, AUROC, average precision score, and AUC-PR.

    Parameters
    ----------
    y_true : Union[np.ndarray, torch.Tensor, list]
        True labels of the data.
    y_pred : Union[np.ndarray, torch.Tensor, list]
        Predicted labels by the model.
    y_prob : Union[np.ndarray, torch.Tensor, list]
        Predicted probabilities for the positive class.

    Returns
    -------
    Dict[str, float]
        Dictionary containing the following metrics:

        - "Balanced Accuracy": Balanced accuracy score
        - "F1 Score": F1 score
        - "Precision": Precision score
        - "Recall": Recall score
        - "AUROC": Area Under the Receiver Operating Characteristic curve
        - "Average Precision Score": Average precision score
        - "AUC-PR": Area Under the Precision-Recall curve
    """
    metrics = {
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "AUROC": roc_auc_score(y_true, y_prob),
        "Average Precision Score": average_precision_score(y_true, y_pred),
    }

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    metrics["AUC-PR"] = auc(recall, precision)

    return metrics
