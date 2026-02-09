#src/eval
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


@dataclass
class Metrics:
    precision_illicit: float
    recall_illicit: float
    f1_illicit: float
    f1_micro: float
    cm: np.ndarray


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0
    )
    f1_micro = f1_score(y_true, y_pred, average="micro")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return Metrics(float(p), float(r), float(f1), float(f1_micro), cm)


def threshold_predictions(p_pos: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return (p_pos >= thr).astype(np.int64)
