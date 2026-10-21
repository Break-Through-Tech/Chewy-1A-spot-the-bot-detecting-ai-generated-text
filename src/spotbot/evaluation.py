"""Shared metrics for human (0) versus AI (1) predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "macro_f1": float(f1.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": pd.DataFrame(
            {"precision": precision, "recall": recall, "f1": f1, "support": support},
            index=pd.Index(["human", "ai"], name="class"),
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


def evaluate_predictions(y_true, y_pred, sources) -> dict:
    """Return overall and per-domain metrics; matrix rows=true, columns=predicted.

    Inputs must use the same row order. Metrics use fixed labels [0, 1],
    including in domains containing only one class. Undefined scores are zero.
    """
    truth, predictions, domains = (np.asarray(values) for values in (y_true, y_pred, sources))
    if any(values.ndim != 1 for values in (truth, predictions, domains)):
        raise ValueError("Labels, predictions, and sources must be one-dimensional")
    if not len(truth) or len(truth) != len(predictions) or len(truth) != len(domains):
        raise ValueError("Labels, predictions, and sources must have the same nonzero length")
    if not np.isin(truth, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Labels and predictions must be 0 (human) or 1 (AI)")
    if pd.isna(domains).any() or not all(isinstance(source, str) and source for source in domains):
        raise ValueError("Each source must be a nonempty domain name")
    result = _metrics(truth, predictions)
    result["per_domain"] = {
        source: _metrics(truth[domains == source], predictions[domains == source])
        for source in sorted(set(domains))
    }
    return result
