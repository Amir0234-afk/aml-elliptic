# src/baselines.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

RFMaxFeatures = float | Literal["sqrt", "log2"]


@dataclass
class BaselineModels:
    lr: LogisticRegression
    rf: RandomForestClassifier


def train_lr(
    X: np.ndarray,
    y: np.ndarray,
    class_weight: Optional[dict[int, float]] = None,
) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=5000,
        class_weight=class_weight,
        solver="lbfgs",
    )
    model.fit(X, y)
    return model


def train_rf(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int,
    max_features: RFMaxFeatures,
    class_weight: Optional[dict[int, float]] = None,
    random_state: int = 42,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def predict_proba_positive(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    return proba[:, 1]
