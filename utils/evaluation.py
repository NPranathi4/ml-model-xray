from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import seaborn as sns  # type: ignore
except Exception:
    sns = None  # type: ignore


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    is_binary: bool = False,
) -> Dict[str, float]:
    average = "binary" if is_binary else "weighted"
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    if is_binary and y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        except Exception:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan

    return metrics


def confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
):
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(2.6, 1.9))
    if sns is not None:
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax,
        )
    else:
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(class_labels)))
        ax.set_yticks(range(len(class_labels)))
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        target_names=class_labels,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})
    return report_df
