from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def counts(labels: np.ndarray) -> dict[int, int]:
    keys, vals = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(keys, vals)}

