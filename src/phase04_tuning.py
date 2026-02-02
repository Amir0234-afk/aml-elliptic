from __future__ import annotations

import json
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterable,TypeVar,cast

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from src.config import ExperimentConfig, Paths
from src.data import class_weights_binary, get_feature_cols
from src.repro import set_seed
from src.runlog import write_run_manifest

T = TypeVar("T")
_SKLEARN_VER = tuple(int(x) for x in sklearn.__version__.split(".")[:2])


try:
    import tqdm.auto as tqdm_auto  # type: ignore
except Exception:  # pragma: no cover
    tqdm_auto = None  # type: ignore


def make_lr(*, C: float, l1_ratio: float, max_iter: int, tol: float, cw: dict[int, float], seed: int) -> LogisticRegression:
    # sklearn >= 1.8: prefer l1_ratio-only API (penalty deprecated)
    if _SKLEARN_VER >= (1, 8):
        return LogisticRegression(
            solver="saga",
            C=C,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol,
            class_weight=cw,
            random_state=seed,
        )
    # sklearn < 1.8: explicit penalty still normal
    pen = "l2" if l1_ratio == 0.0 else ("l1" if l1_ratio == 1.0 else "elasticnet")
    return LogisticRegression(
        solver="saga",
        C=C,
        penalty=pen,
        l1_ratio=None if pen != "elasticnet" else l1_ratio,
        max_iter=max_iter,
        tol=tol,
        class_weight=cw,
        random_state=seed,
    )


def t_iter(it: Iterable[T], **kwargs: Any) -> Iterable[T]:
    """Iterable progress wrapper (safe for Pylance)."""
    if tqdm_auto is None:
        return it
    return cast(Iterable[T], tqdm_auto.tqdm(it, **kwargs))


def t_bar(*, total: int, desc: str, leave: bool = True):
    """Manual progress bar (safe for Pylance)."""
    if tqdm_auto is None:
        class _Dummy:
            def update(self, n: int = 1) -> None: ...
            def set_postfix(self, *a: Any, **k: Any) -> None: ...
            def close(self) -> None: ...
        return _Dummy()
    return tqdm_auto.tqdm(total=total, desc=desc, leave=leave)

# ---- Warning control ----
SUPPRESS_CONVERGENCE_WARNINGS = True
if SUPPRESS_CONVERGENCE_WARNINGS:
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\.linear_model")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\.linear_model")



def load_processed(paths: Paths) -> pd.DataFrame:
    p = paths.processed_dir / "elliptic_labeled.parquet"
    c = paths.processed_dir / "elliptic_labeled.csv"
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError("Processed dataset not found. Run: python -m src.phase01_preprocessing")


def to_binary_labels(y: pd.Series) -> np.ndarray:
    """
    Robust mapping:
      - if values are {1,2} => map 1(illicit)->1, 2(licit)->0
      - if values are {0,1} => keep as-is
    """
    y_int = y.to_numpy(dtype=np.int64)
    uniq = np.unique(y_int)
    if set(uniq.tolist()) <= {0, 1}:
        return y_int.astype(np.int64)
    if set(uniq.tolist()) <= {1, 2}:
        return (y_int == 1).astype(np.int64)
    raise ValueError(f"Unexpected class values: {uniq.tolist()}")


def temporal_train_test_steps(df: pd.DataFrame, train_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    steps = np.sort(df["time_step"].unique().astype(int))
    n_train = int(np.floor(len(steps) * float(train_ratio)))
    n_train = max(1, min(n_train, len(steps) - 1))
    return steps[:n_train], steps[n_train:]


def forward_chaining_cv_splits(
    row_steps: np.ndarray,
    train_steps: np.ndarray,
    *,
    n_splits: int,
    val_steps: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Leakage-safe temporal CV:
      fold k trains on early steps, validates on immediately following val_steps steps.
    Returns row indices into df_train.
    """
    steps = np.sort(np.unique(np.asarray(train_steps, dtype=int)))
    n = int(len(steps))
    if n < 2:
        return []

    val_steps = max(1, int(val_steps))
    n_splits = max(1, int(n_splits))

    max_train_end = n - val_steps
    if max_train_end <= 0:
        return []

    min_train_end = max(1, max_train_end // (n_splits + 1))
    ends = np.unique(np.linspace(min_train_end, max_train_end, num=n_splits, dtype=int))

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for train_end in ends:
        tr_steps = steps[:train_end]
        va_steps = steps[train_end : train_end + val_steps]
        if tr_steps.size == 0 or va_steps.size == 0:
            continue

        tr_idx = np.flatnonzero(np.isin(row_steps, tr_steps))
        va_idx = np.flatnonzero(np.isin(row_steps, va_steps))
        if tr_idx.size == 0 or va_idx.size == 0:
            continue
        splits.append((tr_idx, va_idx))
    return splits


def best_threshold_for_f1(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """
    Find threshold maximizing illicit F1 (pos_label=1).
    score should be probability of class 1 (preferred).
    """
    s = np.asarray(score, dtype=float)
    y = np.asarray(y_true, dtype=int)
    if s.size == 0:
        return 0.5, -1.0

    uniq = np.unique(s)
    if uniq.size > 128:
        qs = np.linspace(0.0, 1.0, 513)
        uniq = np.unique(np.quantile(s, qs))

    best_t = 0.5
    best_f1 = -1.0
    for t in uniq:
        pred = (s >= t).astype(int)
        f1 = float(f1_score(y, pred, pos_label=1, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def _penalty_name_from_l1_ratio(l1_ratio: float) -> str:
    if abs(l1_ratio - 0.0) < 1e-12:
        return "l2"
    if abs(l1_ratio - 1.0) < 1e-12:
        return "l1"
    return "elasticnet"


def main() -> None:
    paths = Paths()
    cfg = ExperimentConfig()
    set_seed(cfg.seed, deterministic_torch=False)

    (paths.results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "model_artifacts").mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "logs").mkdir(parents=True, exist_ok=True)

    write_run_manifest(
        paths.results_dir / "logs",
        phase="04",
        cfg=cfg,
        data_files=[
            paths.processed_dir / "elliptic_labeled.parquet",
            paths.processed_dir / "elliptic_labeled.csv",
        ],
        extra={"note": "phase 04: forward-chaining CV tuning for tabular models + threshold tuning (no test leakage)"},
    )

    df = load_processed(paths)
    df = df.sort_values(["time_step", "txId"]).reset_index(drop=True)

    feature_mode: str = "AF"
    feat_cols = get_feature_cols(df, feature_mode)

    train_steps, test_steps = temporal_train_test_steps(df, cfg.train_ratio)
    df_train = df[df["time_step"].isin(train_steps)].copy()
    df_test = df[df["time_step"].isin(test_steps)].copy()

    X_train_raw = df_train[feat_cols].to_numpy(dtype=np.float32)
    y_train = to_binary_labels(df_train["class"])
    X_test_raw = df_test[feat_cols].to_numpy(dtype=np.float32)
    y_test = to_binary_labels(df_test["class"])

    if np.unique(y_train).size < 2:
        raise RuntimeError("Training window contains only one class; cannot tune/train classifiers.")

    row_steps = df_train["time_step"].to_numpy(dtype=int)
    splits = forward_chaining_cv_splits(
        row_steps=row_steps,
        train_steps=train_steps,
        n_splits=int(cfg.tabular_cv_splits),
        val_steps=int(cfg.tabular_cv_val_steps),
    )
    if not splits:
        raise RuntimeError("Could not create temporal CV splits; reduce val_steps or increase train window.")

    cw = class_weights_binary(y_train)  # {0,1}
    rng = np.random.default_rng(cfg.seed)

    n_trials = max(1, int(cfg.tabular_tune_trials))
    n_folds = len(splits)

    # ------------------- Logistic Regression (NO penalty/n_jobs; uses l1_ratio only) -------------------
    C_grid = np.logspace(-3, 2, 24).astype(float)

    # l1_ratio encodes penalty type in sklearn 1.8+:
    #   0.0 -> l2, 1.0 -> l1, (0,1) -> elasticnet
    l1_ratio_grid = np.array([0.0, 1.0, 0.15, 0.30, 0.50, 0.70, 0.85], dtype=float)

    lr_trials: List[Dict[str, Any]] = []
    best_lr: Dict[str, Any] | None = None
    best_lr_score = -1.0

    total_lr_fits = n_trials * n_folds
    pbar_lr = t_bar(total=total_lr_fits, desc=f"LR HPO ({n_trials}×{len(splits)} folds)", leave=True)


    for _ in range(n_trials):
        C = float(rng.choice(C_grid))
        l1r = float(rng.choice(l1_ratio_grid))

        fold_scores: List[float] = []
        for tr_idx, va_idx in splits:
            # Fit scaler only on fold-train to avoid leakage into fold-val
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X_train_raw[tr_idx])
            Xva = scaler.transform(X_train_raw[va_idx])

            model = LogisticRegression(
                solver="saga",
                C=C,
                l1_ratio=l1r,          # replaces penalty in sklearn 1.8+
                max_iter=int(getattr(cfg, "lr_max_iter", 5000)),
                tol=float(getattr(cfg, "lr_tol", 1e-4)),
                class_weight=cw,
                random_state=cfg.seed,
            )
            model.fit(Xtr, y_train[tr_idx])

            proba = model.predict_proba(Xva)[:, 1]
            _, f1v = best_threshold_for_f1(y_train[va_idx], proba)
            fold_scores.append(float(f1v))

            pbar_lr.update(1)

        cv_f1 = float(np.mean(fold_scores)) if fold_scores else -1.0
        trial = {
            "C": C,
            "l1_ratio": l1r,
            "penalty_name": _penalty_name_from_l1_ratio(l1r),
            "cv_f1_illicit": cv_f1,
        }
        lr_trials.append(trial)

        if cv_f1 > best_lr_score:
            best_lr_score = cv_f1
            best_lr = trial

        pbar_lr.set_postfix(
            best=f"{best_lr_score:.4f}",
            last=f"{cv_f1:.4f}",
            C=f"{C:.3g}",
            l1r=f"{l1r:.2f}",
            pen=_penalty_name_from_l1_ratio(l1r),
        )

    pbar_lr.close()

    if best_lr is None:
        raise RuntimeError("LR tuning failed.")

    # Threshold tuning: last val_steps of TRAIN steps is tail (no test leakage)
    train_steps_sorted = np.sort(np.unique(train_steps.astype(int)))
    n_tail = max(1, int(cfg.tabular_cv_val_steps))
    tail_steps = train_steps_sorted[-n_tail:]
    base_steps = train_steps_sorted[:-n_tail] if train_steps_sorted.size > n_tail else train_steps_sorted[:-1]

    if base_steps.size == 0:
        base_steps = train_steps_sorted[:-1]
        tail_steps = train_steps_sorted[-1:]

    df_base = df_train[df_train["time_step"].isin(base_steps)].copy()
    df_tail = df_train[df_train["time_step"].isin(tail_steps)].copy()

    X_base_raw = df_base[feat_cols].to_numpy(dtype=np.float32)
    y_base = to_binary_labels(df_base["class"])
    X_tail_raw = df_tail[feat_cols].to_numpy(dtype=np.float32)
    y_tail = to_binary_labels(df_tail["class"])

    # Fit scaler on base only (no tail leakage)
    scaler_final = StandardScaler()
    X_base = scaler_final.fit_transform(X_base_raw)
    X_tail = scaler_final.transform(X_tail_raw)

    lr_final = LogisticRegression(
        solver="saga",
        C=float(best_lr["C"]),
        l1_ratio=float(best_lr["l1_ratio"]),
        max_iter=int(getattr(cfg, "lr_max_iter", 5000)),
        tol=float(getattr(cfg, "lr_tol", 1e-4)),
        class_weight=cw,
        random_state=cfg.seed,
    )
    lr_final.fit(X_base, y_base)
    tail_proba = lr_final.predict_proba(X_tail)[:, 1]
    lr_thr, lr_tail_f1 = best_threshold_for_f1(y_tail, tail_proba)

    # Refit LR on ALL train with scaler fit on ALL train (artifact you will use later)
    scaler_all = StandardScaler()
    X_train_all = scaler_all.fit_transform(X_train_raw)
    X_test_all = scaler_all.transform(X_test_raw)

    lr_final.fit(X_train_all, y_train)
    test_proba = lr_final.predict_proba(X_test_all)[:, 1]
    yhat_lr = (test_proba >= lr_thr).astype(int)

    lr_test_f1 = float(f1_score(y_test, yhat_lr, pos_label=1, zero_division=0))
    lr_test_prec = float(precision_score(y_test, yhat_lr, pos_label=1, zero_division=0))
    lr_test_rec = float(recall_score(y_test, yhat_lr, pos_label=1, zero_division=0))

    # ------------------- Random Forest (raw) -------------------
    rf_trials: List[Dict[str, Any]] = []
    best_rf: Dict[str, Any] | None = None
    best_rf_score = -1.0

    n_estimators_grid = np.array([200, 400, 600, 800, 1000], dtype=int)
    max_features_grid: List[float | str] = ["sqrt", "log2", 0.3, 0.5, 0.7]
    max_depth_grid: List[int | None] = [None, 10, 20, 30, 40]
    min_samples_split_grid = np.array([2, 5, 10], dtype=int)
    min_samples_leaf_grid = np.array([1, 2, 5, 10], dtype=int)
    max_samples_grid: List[float | None] = [None, 0.7, 0.85, 0.95]

    def _pick(items: list[Any]) -> Any:
        return items[int(rng.integers(0, len(items)))]

    total_rf_fits = n_trials * n_folds
    pbar_rf = t_bar(total=total_rf_fits, desc=f"RF HPO ({n_trials}×{len(splits)} folds)", leave=True)

    for _ in range(n_trials):
        params = {
            "n_estimators": int(rng.choice(n_estimators_grid)),
            "max_features": _pick(max_features_grid),
            "max_depth": _pick(max_depth_grid),
            "min_samples_split": int(rng.choice(min_samples_split_grid)),
            "min_samples_leaf": int(rng.choice(min_samples_leaf_grid)),
            "max_samples": _pick(max_samples_grid),
        }

        fold_scores: List[float] = []
        for tr_idx, va_idx in splits:
            model = RandomForestClassifier(
                n_estimators=int(params["n_estimators"]),
                max_features=params["max_features"],
                max_depth=params["max_depth"],
                min_samples_split=int(params["min_samples_split"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                max_samples=params["max_samples"],
                class_weight=cw,
                n_jobs=-1,  # RF still uses this; not deprecated here
                random_state=cfg.seed,
            )
            model.fit(X_train_raw[tr_idx], y_train[tr_idx])
            yhat = model.predict(X_train_raw[va_idx])
            f1v = float(f1_score(y_train[va_idx], yhat, pos_label=1, zero_division=0))
            fold_scores.append(f1v)

            pbar_rf.update(1)

        cv_f1 = float(np.mean(fold_scores)) if fold_scores else -1.0
        trial = {**params, "cv_f1_illicit": cv_f1}
        rf_trials.append(trial)

        if cv_f1 > best_rf_score:
            best_rf_score = cv_f1
            best_rf = trial

        pbar_rf.set_postfix(best=f"{best_rf_score:.4f}", last=f"{cv_f1:.4f}", n=params["n_estimators"])

    pbar_rf.close()

    if best_rf is None:
        raise RuntimeError("RF tuning failed.")

    rf_final = RandomForestClassifier(
        n_estimators=int(best_rf["n_estimators"]),
        max_features=best_rf["max_features"],
        max_depth=best_rf["max_depth"],
        min_samples_split=int(best_rf["min_samples_split"]),
        min_samples_leaf=int(best_rf["min_samples_leaf"]),
        max_samples=best_rf["max_samples"],
        class_weight=cw,
        n_jobs=-1,
        random_state=cfg.seed,
    )
    rf_final.fit(X_train_raw, y_train)
    yhat_rf = rf_final.predict(X_test_raw)

    rf_test_f1 = float(f1_score(y_test, yhat_rf, pos_label=1, zero_division=0))
    rf_test_prec = float(precision_score(y_test, yhat_rf, pos_label=1, zero_division=0))
    rf_test_rec = float(recall_score(y_test, yhat_rf, pos_label=1, zero_division=0))

    # ---- Save tuned artifacts ----
    base = paths.results_dir / "model_artifacts"
    joblib.dump(lr_final, base / "tuned_lr_AF.joblib")
    joblib.dump(scaler_all, base / "tuned_lr_AF_scaler.joblib")
    joblib.dump({"threshold": float(lr_thr)}, base / "tuned_lr_AF_threshold.joblib")
    joblib.dump(rf_final, base / "tuned_rf_AF.joblib")

    report = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "feature_mode": feature_mode,
        "split": {"train_steps": train_steps.tolist(), "test_steps": test_steps.tolist()},
        "cv": {"n_splits": int(cfg.tabular_cv_splits), "val_steps": int(cfg.tabular_cv_val_steps), "folds": int(len(splits))},
        "class_weights_train": {str(k): float(v) for k, v in cw.items()},
        "logreg": {
            "best_hparams": {
                "C": float(best_lr["C"]),
                "l1_ratio": float(best_lr["l1_ratio"]),
                "penalty_name": str(best_lr["penalty_name"]),
            },
            "best_cv_f1_illicit": float(best_lr_score),
            "threshold_tuned_on_tail_steps": tail_steps.tolist(),
            "threshold": float(lr_thr),
            "tail_f1_illicit": float(lr_tail_f1),
            "test": {"f1_illicit": lr_test_f1, "precision_illicit": lr_test_prec, "recall_illicit": lr_test_rec},
            "trials_top10": sorted(lr_trials, key=lambda d: d["cv_f1_illicit"], reverse=True)[:10],
        },
        "random_forest": {
            "best_hparams": {
                "n_estimators": int(best_rf["n_estimators"]),
                "max_features": best_rf["max_features"],
                "max_depth": best_rf["max_depth"],
                "min_samples_split": int(best_rf["min_samples_split"]),
                "min_samples_leaf": int(best_rf["min_samples_leaf"]),
                "max_samples": best_rf["max_samples"],
            },
            "best_cv_f1_illicit": float(best_rf_score),
            "test": {"f1_illicit": rf_test_f1, "precision_illicit": rf_test_prec, "recall_illicit": rf_test_rec},
            "trials_top10": sorted(rf_trials, key=lambda d: d["cv_f1_illicit"], reverse=True)[:10],
        },
    }

    (paths.results_dir / "metrics" / "phase04_tuning_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
