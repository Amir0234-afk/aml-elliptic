# src/baselines.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.tabular_backend import get_tabular_backend

RFMaxFeatures = float | Literal["sqrt", "log2"]


@dataclass
class BaselineModels:
    lr: LogisticRegression
    rf: RandomForestClassifier


def train_lr(
    X: np.ndarray,
    y: np.ndarray,
    class_weight: Optional[dict[int, float]] = None,
    *,
    seed: int = 23,
):
    backend = get_tabular_backend()
    return backend.train_lr(X, y, class_weight=class_weight, seed=seed)


def train_rf(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int,
    max_features: RFMaxFeatures,
    class_weight: Optional[dict[int, float]] = None,
    random_state: int = 23,
):
    backend = get_tabular_backend()
    return backend.train_rf(
        X,
        y,
        n_estimators=n_estimators,
        max_features=max_features,
        class_weight=class_weight,
        seed=random_state,
    )


def predict_proba_positive(model, X: np.ndarray) -> np.ndarray:
    backend = get_tabular_backend()
    return backend.predict_proba_positive(model, X)
