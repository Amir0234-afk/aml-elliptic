from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import ExperimentConfig, Paths
from src.data import make_tabular_split, class_weights_binary


def load_processed(paths: Paths) -> pd.DataFrame:
    p = paths.processed_dir / "elliptic_labeled.parquet"
    c = paths.processed_dir / "elliptic_labeled.csv"
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError("Processed dataset not found. Run: python -m src.phase01_preprocessing")


def main() -> None:
    paths = Paths()
    cfg = ExperimentConfig()
    (paths.results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "model_artifacts").mkdir(parents=True, exist_ok=True)

    df = load_processed(paths)

    # Temporal split (we tune on validation slice inside train steps by reusing make_tabular_split)
    split = make_tabular_split(df, "AF", train_ratio=cfg.train_ratio, normalize=True)

    # Create a validation split from the tail of training steps (same policy as graph build)
    train_steps = split.train_steps
    unique_train = np.unique(train_steps)
    n_val_steps = max(1, int(len(unique_train) * cfg.val_ratio_within_train))
    val_steps = unique_train[-n_val_steps:]
    tr_steps = unique_train[:-n_val_steps]

    df_tr = df[df["time_step"].isin(tr_steps)].copy()
    df_va = df[df["time_step"].isin(val_steps)].copy()

    # Build arrays (AF)
    tr = make_tabular_split(df_tr, "AF", train_ratio=1.0 - 1e-9, normalize=True)  # all as "train"
    va = make_tabular_split(df_va, "AF", train_ratio=1.0 - 1e-9, normalize=True)

    cw = class_weights_binary(tr.y_train)

    # ---- Tune LogisticRegression(C) ----
    lr_grid = {"C": [0.1, 0.3, 1.0, 3.0, 10.0]}
    best_lr = None
    best_lr_f1 = -1.0
    best_lr_params = None

    for params in ParameterGrid(lr_grid):
        model = LogisticRegression(
            max_iter=5000,
            class_weight=cw,
            solver="lbfgs",
            C=params["C"],
        )
        model.fit(tr.X_train, tr.y_train)
        yhat = model.predict(va.X_test)  # va uses its own scaler; consistent since both are standardized
        f1 = f1_score(va.y_test, yhat, pos_label=1)
        if f1 > best_lr_f1:
            best_lr_f1 = float(f1)
            best_lr = model
            best_lr_params = params

    # ---- Tune RandomForest ----
    rf_grid = {
        "n_estimators": [200, 400],
        "max_features": ["sqrt", "log2"],
        "max_depth": [None, 12, 24],
        "min_samples_split": [2, 5],
    }
    best_rf = None
    best_rf_f1 = -1.0
    best_rf_params = None

    # RF should use raw features (no normalization). Rebuild raw AF arrays:
    tr_raw = make_tabular_split(df_tr, "AF", train_ratio=1.0 - 1e-9, normalize=False)
    va_raw = make_tabular_split(df_va, "AF", train_ratio=1.0 - 1e-9, normalize=False)

    for params in ParameterGrid(rf_grid):
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_features=params["max_features"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            class_weight=cw,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(tr_raw.X_train, tr_raw.y_train)
        yhat = model.predict(va_raw.X_test)
        f1 = f1_score(va_raw.y_test, yhat, pos_label=1)
        if f1 > best_rf_f1:
            best_rf_f1 = float(f1)
            best_rf = model
            best_rf_params = params

    # Save tuned artifacts
    base = paths.results_dir / "model_artifacts"
    joblib.dump(best_lr, base / "tuned_lr_AF.joblib")
    joblib.dump(best_rf, base / "tuned_rf_AF.joblib")

    report = {
        "tuned_on": {"feature_mode": "AF", "val_steps": val_steps.tolist(), "train_steps_used": tr_steps.tolist()},
        "logreg": {"best_params": best_lr_params, "val_f1_illicit": best_lr_f1},
        "random_forest": {"best_params": best_rf_params, "val_f1_illicit": best_rf_f1},
    }
    (paths.results_dir / "metrics" / "phase04_tuning_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
