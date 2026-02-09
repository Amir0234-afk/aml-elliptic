# src/phase04_tuning.py
from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve

from src.config import ExperimentConfig, Paths
from src.data import FeatureMode, class_weights_binary, get_feature_cols
from src.eval import compute_metrics, threshold_predictions
from src.runlog import write_run_manifest
from src.tabular_backend import TabularBackend, get_tabular_backend
from src.viz import (
    save_pr_curve,
    save_threshold_sweep,
    save_calibration_curve,
    save_confusion_matrix_heatmap,
)
from src.repro import set_seed

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x: Iterable[Any], **kwargs: Any):  # type: ignore
        return x


def _ensure_dirs(paths: Paths) -> None:
    (paths.results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "logs").mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "model_artifacts" / "phase04").mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "visualizations" / "phase04").mkdir(parents=True, exist_ok=True)


def _load_labeled(paths: Paths) -> pd.DataFrame:
    p = paths.processed_dir / "elliptic_labeled.parquet"
    c = paths.processed_dir / "elliptic_labeled.csv"
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError("Labeled dataset not found. Run: python -m src.phase01_preprocessing")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_safe(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def _temporal_train_test_steps(df_labeled: pd.DataFrame, train_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    steps = np.sort(df_labeled["time_step"].unique().astype(int))
    n_train = int(np.floor(len(steps) * float(train_ratio)))
    n_train = max(1, min(n_train, len(steps) - 1))
    return steps[:n_train], steps[n_train:]


def _forward_chaining_cv_splits(
    row_steps: np.ndarray,
    train_steps: np.ndarray,
    *,
    n_splits: int,
    val_steps: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Leakage-safe temporal CV:
      fold k trains on early steps, validates on immediately following val_steps steps.
    Returns row indices into df_train-aligned arrays.
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


def _best_threshold_for_f1(y_true: np.ndarray, p_pos: np.ndarray) -> tuple[float, float]:
    """
    Find threshold maximizing illicit F1 (pos_label=1). Uses unique probs; quantiles if too many.
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(p_pos, dtype=float)
    if s.size == 0:
        return 0.5, -1.0

    uniq = np.unique(s)
    if uniq.size > 256:
        qs = np.linspace(0.0, 1.0, 513)
        uniq = np.unique(np.quantile(s, qs))

    best_t = 0.5
    best_f1 = -1.0
    for t in uniq:
        pred = threshold_predictions(s, float(t))
        m = compute_metrics(y, pred)
        if m.f1_illicit > best_f1:
            best_f1 = float(m.f1_illicit)
            best_t = float(t)
    return float(best_t), float(best_f1)


def _threshold_sweep(y_true: np.ndarray, p_pos: np.ndarray, thresholds: np.ndarray) -> dict[str, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(p_pos, dtype=float)
    t = np.asarray(thresholds, dtype=float)

    f1 = np.zeros_like(t, dtype=float)
    pr = np.zeros_like(t, dtype=float)
    rc = np.zeros_like(t, dtype=float)

    for i, thr in enumerate(t):
        pred = threshold_predictions(s, float(thr))
        m = compute_metrics(y, pred)
        f1[i] = float(m.f1_illicit)
        pr[i] = float(m.precision_illicit)
        rc[i] = float(m.recall_illicit)

    return {"thresholds": t, "f1": f1, "precision": pr, "recall": rc}


def _tail_split_within_train(train_steps: np.ndarray, tail_steps_count: int) -> tuple[np.ndarray, np.ndarray]:
    ts = np.sort(np.unique(train_steps.astype(int)))
    n_tail = max(1, int(tail_steps_count))
    if ts.size <= 1:
        raise RuntimeError("Train window too small to create a tail split for threshold tuning.")
    if ts.size <= n_tail:
        n_tail = 1
    tail = ts[-n_tail:]
    base = ts[:-n_tail]
    if base.size == 0:
        base = ts[:-1]
        tail = ts[-1:]
    return base, tail


def _standardize_fit_transform(X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    Xn = (X - mu) / sd
    return Xn, {"mu": mu, "sd": sd}


def _standardize_transform(X: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    return (X - params["mu"]) / params["sd"]


def _save_artifact_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(obj), indent=2), encoding="utf-8")


def _persist_artifact_inventory(paths: Paths, backend_name: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    inv_path = paths.results_dir / "logs" / f"phase04_artifacts_{backend_name}_{stamp}.json"
    roots = [
        paths.results_dir / "metrics",
        paths.results_dir / "visualizations" / "phase04",
        paths.results_dir / "model_artifacts" / "phase04",
    ]

    artifacts: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*"), key=lambda q: str(q)):
            if not p.is_file():
                continue
            if p.suffix.lower() in {".json", ".png", ".joblib"}:
                artifacts.append({"path": str(p), "bytes": int(p.stat().st_size), "sha256": _sha256_file(p)})

    inv = {
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "04",
        "tabular_backend": backend_name,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    inv_path.write_text(json.dumps(inv, indent=2), encoding="utf-8")


def _tune_lr_C(
    *,
    backend: TabularBackend,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    cw: dict[int, float],
    cfg: ExperimentConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    C_grid = np.logspace(-3, 2, num=24).astype(float)
    trials = min(int(cfg.tabular_tune_trials), int(C_grid.size))
    cand_C = rng.choice(C_grid, size=trials, replace=False)

    best = {"C": 1.0, "cv_f1": -1.0}
    trial_rows: list[dict[str, Any]] = []

    for C in tqdm(cand_C, desc="LR HPO (C)", leave=False):
        fold_f1: list[float] = []
        fold_thr: list[float] = []
        for tr_idx, va_idx in splits:
            Xtr = X_train_raw[tr_idx]
            Xva = X_train_raw[va_idx]

            Xtr_n, ss = _standardize_fit_transform(Xtr)
            Xva_n = _standardize_transform(Xva, ss)

            model = backend.train_lr(
                Xtr_n.astype(np.float32, copy=False),
                y_train[tr_idx],
                C=float(C),
                max_iter=int(cfg.lr_max_iter),
                tol=float(cfg.lr_tol),
                class_weight=cw,
                seed=int(cfg.seed),
            )
            pva = backend.predict_proba_positive(model, Xva_n.astype(np.float32, copy=False))
            thr, f1v = _best_threshold_for_f1(y_train[va_idx], pva)
            fold_f1.append(float(f1v))
            fold_thr.append(float(thr))

        cv_f1 = float(np.mean(fold_f1)) if fold_f1 else -1.0
        row = {"C": float(C), "cv_f1_illicit": cv_f1, "thr_mean": float(np.mean(fold_thr)) if fold_thr else 0.5}
        trial_rows.append(row)

        if cv_f1 > float(best["cv_f1"]):
            best = {"C": float(C), "cv_f1": cv_f1}

    return {"best": best, "trials_top10": sorted(trial_rows, key=lambda d: d["cv_f1_illicit"], reverse=True)[:10]}


def _tune_rf_params(
    *,
    backend: TabularBackend,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    cw: dict[int, float],
    cfg: ExperimentConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    # Backend-aware grids (mirrors your phase03 intent)
    if backend.name == "cuml":
        max_features_grid: list[Any] = ["sqrt", 0.3, 0.5]
        max_depth_grid: list[Any] = [10, 16, 20, 30, 40]  # avoid None for cuML stability
        min_leaf_grid = np.array([1], dtype=int)
        min_split_grid = np.array([2], dtype=int)
        max_samples_grid: list[Any] = [None]
    else:
        max_features_grid = ["sqrt", "log2", 0.3, 0.5, 0.7]
        max_depth_grid = [None, 10, 20, 30, 40]
        min_leaf_grid = np.array([1, 2, 5, 10], dtype=int)
        min_split_grid = np.array([2, 5, 10], dtype=int)
        max_samples_grid = [None, 0.7, 0.85, 0.95]

    n_estimators_grid = np.array([200, 400, 600, 800, 1000], dtype=int)
    n_trials = max(1, int(cfg.tabular_tune_trials))

    best: dict[str, Any] = {
        "n_estimators": int(cfg.rf_n_estimators),
        "max_features": cfg.rf_max_features,
        "max_depth": None if backend.name != "cuml" else 16,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "max_samples": None,
        "cv_f1": -1.0,
    }
    best_rows: list[dict[str, Any]] = []

    def pick(items: list[Any]) -> Any:
        return items[int(rng.integers(0, len(items)))]

    for _ in tqdm(range(n_trials), desc="RF HPO", leave=False):
        params = {
            "n_estimators": int(rng.choice(n_estimators_grid)),
            "max_features": pick(max_features_grid),
            "max_depth": pick(max_depth_grid),
            "min_samples_leaf": int(rng.choice(min_leaf_grid)),
            "min_samples_split": int(rng.choice(min_split_grid)),
            "max_samples": pick(max_samples_grid),
        }

        fold_f1: list[float] = []
        for tr_idx, va_idx in splits:
            model = backend.train_rf(
                X_train_raw[tr_idx],
                y_train[tr_idx],
                n_estimators=int(params["n_estimators"]),
                max_features=params["max_features"],
                max_depth=cast(int | None, params["max_depth"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                min_samples_split=int(params["min_samples_split"]),
                max_samples=cast(float | None, params["max_samples"]),
                class_weight=cw,
                seed=int(cfg.seed),
            )
            pva = backend.predict_proba_positive(model, X_train_raw[va_idx])
            _, f1v = _best_threshold_for_f1(y_train[va_idx], pva)
            fold_f1.append(float(f1v))

        cv_f1 = float(np.mean(fold_f1)) if fold_f1 else -1.0
        row = {**params, "cv_f1_illicit": cv_f1}
        best_rows.append(row)

        if cv_f1 > float(best["cv_f1"]):
            best = {**params, "cv_f1": cv_f1}

    return {"best": best, "trials_top10": sorted(best_rows, key=lambda d: d["cv_f1_illicit"], reverse=True)[:10]}


def _run_mode(
    *,
    df: pd.DataFrame,
    feature_mode: FeatureMode,
    backend: TabularBackend,
    cfg: ExperimentConfig,
    paths: Paths,
    rng: np.random.Generator,
) -> dict[str, Any]:
    df = df.sort_values(["time_step", "txId"]).reset_index(drop=True)

    train_steps, test_steps = _temporal_train_test_steps(df, cfg.train_ratio)
    feat_cols = get_feature_cols(df, feature_mode)

    df_train = df[df["time_step"].isin(train_steps)].copy()
    df_test = df[df["time_step"].isin(test_steps)].copy()

    X_train_raw = df_train[feat_cols].to_numpy(dtype=np.float32)
    X_test_raw = df_test[feat_cols].to_numpy(dtype=np.float32)
    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_raw = np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)

    y_train = df_train["class"].to_numpy(dtype=np.int64)
    y_test = df_test["class"].to_numpy(dtype=np.int64)

    uniq = set(np.unique(y_train).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"Phase04 expects labeled class in {{0,1}}. Got train classes={sorted(list(uniq))}.")

    cw = class_weights_binary(y_train)
    row_steps = df_train["time_step"].to_numpy(dtype=int)
    splits = _forward_chaining_cv_splits(
        row_steps=row_steps,
        train_steps=train_steps,
        n_splits=int(cfg.tabular_cv_splits),
        val_steps=int(cfg.tabular_cv_val_steps),
    )
    if not splits:
        raise RuntimeError("Could not create temporal CV splits (try smaller tabular_cv_val_steps).")

    # Tail split inside TRAIN for threshold tuning (no test leakage)
    base_steps, tail_steps = _tail_split_within_train(train_steps, int(cfg.tabular_cv_val_steps))
    df_base = df_train[df_train["time_step"].isin(base_steps)].copy()
    df_tail = df_train[df_train["time_step"].isin(tail_steps)].copy()

    X_base_raw = df_base[feat_cols].to_numpy(dtype=np.float32)
    X_tail_raw = df_tail[feat_cols].to_numpy(dtype=np.float32)
    X_base_raw = np.nan_to_num(X_base_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_tail_raw = np.nan_to_num(X_tail_raw, nan=0.0, posinf=0.0, neginf=0.0)
    y_base = df_base["class"].to_numpy(dtype=np.int64)
    y_tail = df_tail["class"].to_numpy(dtype=np.int64)

    out_dir_viz = paths.results_dir / "visualizations" / "phase04"
    art_dir = paths.results_dir / "model_artifacts" / "phase04"

    report: dict[str, Any] = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "feature_mode": feature_mode,
        "tabular_backend": backend.name,
        "split": {"train_steps": train_steps.tolist(), "test_steps": test_steps.tolist()},
        "threshold_tune_tail_steps": tail_steps.tolist(),
        "class_weights_train": {str(k): float(v) for k, v in cw.items()},
        "cv": {"n_splits": int(cfg.tabular_cv_splits), "val_steps": int(cfg.tabular_cv_val_steps), "folds": int(len(splits))},
    }

    # ---------------- LR (C tuning + threshold) ----------------
    lr_hpo = _tune_lr_C(
        backend=backend,
        X_train_raw=X_train_raw,
        y_train=y_train,
        splits=splits,
        cw=cw,
        cfg=cfg,
        rng=rng,
    )
    best_lr_C = float(lr_hpo["best"]["C"])

    # Threshold tune on tail (train on base only)
    Xb_n, ss_base = _standardize_fit_transform(X_base_raw)
    Xt_n = _standardize_transform(X_tail_raw, ss_base)
    lr_tail = backend.train_lr(
        Xb_n.astype(np.float32, copy=False),
        y_base,
        C=best_lr_C,
        max_iter=int(cfg.lr_max_iter),
        tol=float(cfg.lr_tol),
        class_weight=cw,
        seed=int(cfg.seed),
    )
    p_tail = backend.predict_proba_positive(lr_tail, Xt_n.astype(np.float32, copy=False))
    lr_thr, lr_tail_f1 = _best_threshold_for_f1(y_tail, p_tail)

    # Final LR: fit scaler on ALL train, fit on ALL train, eval on test
    Xtr_n, ss_all = _standardize_fit_transform(X_train_raw)
    Xte_n = _standardize_transform(X_test_raw, ss_all)

    lr_final = backend.train_lr(
        Xtr_n.astype(np.float32, copy=False),
        y_train,
        C=best_lr_C,
        max_iter=int(cfg.lr_max_iter),
        tol=float(cfg.lr_tol),
        class_weight=cw,
        seed=int(cfg.seed),
    )
    p_test = backend.predict_proba_positive(lr_final, Xte_n.astype(np.float32, copy=False))
    yhat_lr = threshold_predictions(p_test, lr_thr)
    m_lr = compute_metrics(y_test, yhat_lr)

    # Visuals (LR)
    prec, rec, _ = precision_recall_curve(y_test, p_test)
    save_pr_curve(prec, rec, out_dir_viz, fname=f"pr_lr_{feature_mode}_{backend.name}.png", title=f"PR curve (LR, {feature_mode}, {backend.name})")

    sweep_t = np.linspace(0.0, 1.0, 201, dtype=float)
    sweep = _threshold_sweep(y_tail, p_tail, sweep_t)
    save_threshold_sweep(
        sweep["thresholds"], sweep["f1"], sweep["precision"], sweep["recall"],
        out_dir_viz,
        fname=f"thr_sweep_lr_{feature_mode}_{backend.name}.png",
        title=f"Threshold sweep on tail (LR, {feature_mode}, {backend.name})",
    )

    pt, pp = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")
    save_calibration_curve(pt, pp, out_dir_viz, fname=f"calib_lr_{feature_mode}_{backend.name}.png", title=f"Calibration (LR, {feature_mode}, {backend.name})")

    save_confusion_matrix_heatmap(m_lr.cm, out_dir_viz, fname=f"cm_lr_{feature_mode}_{backend.name}.png", title=f"Confusion matrix (LR, {feature_mode}, {backend.name})", normalize=False)
    save_confusion_matrix_heatmap(m_lr.cm, out_dir_viz, fname=f"cm_lr_{feature_mode}_{backend.name}_norm.png", title=f"Confusion matrix (LR, {feature_mode}, {backend.name})", normalize=True)

    # Persist LR artifacts
    joblib.dump(lr_final, art_dir / f"lr_{feature_mode}_{backend.name}.joblib")
    joblib.dump({"mu": ss_all["mu"], "sd": ss_all["sd"]}, art_dir / f"lr_scaler_{feature_mode}.joblib")
    _save_artifact_json(art_dir / f"lr_threshold_{feature_mode}_{backend.name}.json", {"threshold": float(lr_thr)})

    report["logreg"] = {
        "best_hparams": {"C": float(best_lr_C)},
        "best_cv_f1_illicit": float(lr_hpo["best"]["cv_f1"]),
        "threshold": float(lr_thr),
        "tail_f1_illicit": float(lr_tail_f1),
        "test": m_lr.__dict__,
        "trials_top10": lr_hpo["trials_top10"],
    }

    # ---------------- RF (param tuning + threshold) ----------------
    rf_hpo = _tune_rf_params(
        backend=backend,
        X_train_raw=X_train_raw,
        y_train=y_train,
        splits=splits,
        cw=cw,
        cfg=cfg,
        rng=rng,
    )
    best_rf = cast(dict[str, Any], rf_hpo["best"])

    rf_tail = backend.train_rf(
        X_base_raw,
        y_base,
        n_estimators=int(best_rf["n_estimators"]),
        max_features=best_rf["max_features"],
        max_depth=cast(int | None, best_rf["max_depth"]),
        min_samples_leaf=int(best_rf["min_samples_leaf"]),
        min_samples_split=int(best_rf["min_samples_split"]),
        max_samples=cast(float | None, best_rf["max_samples"]),
        class_weight=cw,
        seed=int(cfg.seed),
    )
    p_tail_rf = backend.predict_proba_positive(rf_tail, X_tail_raw)
    rf_thr, rf_tail_f1 = _best_threshold_for_f1(y_tail, p_tail_rf)

    rf_final = backend.train_rf(
        X_train_raw,
        y_train,
        n_estimators=int(best_rf["n_estimators"]),
        max_features=best_rf["max_features"],
        max_depth=cast(int | None, best_rf["max_depth"]),
        min_samples_leaf=int(best_rf["min_samples_leaf"]),
        min_samples_split=int(best_rf["min_samples_split"]),
        max_samples=cast(float | None, best_rf["max_samples"]),
        class_weight=cw,
        seed=int(cfg.seed),
    )
    p_test_rf = backend.predict_proba_positive(rf_final, X_test_raw)
    yhat_rf = threshold_predictions(p_test_rf, rf_thr)
    m_rf = compute_metrics(y_test, yhat_rf)

    # Visuals (RF)
    prec_rf, rec_rf, _ = precision_recall_curve(y_test, p_test_rf)
    save_pr_curve(prec_rf, rec_rf, out_dir_viz, fname=f"pr_rf_{feature_mode}_{backend.name}.png", title=f"PR curve (RF, {feature_mode}, {backend.name})")

    sweep_rf = _threshold_sweep(y_tail, p_tail_rf, sweep_t)
    save_threshold_sweep(
        sweep_rf["thresholds"], sweep_rf["f1"], sweep_rf["precision"], sweep_rf["recall"],
        out_dir_viz,
        fname=f"thr_sweep_rf_{feature_mode}_{backend.name}.png",
        title=f"Threshold sweep on tail (RF, {feature_mode}, {backend.name})",
    )

    pt_rf, pp_rf = calibration_curve(y_test, p_test_rf, n_bins=10, strategy="quantile")
    save_calibration_curve(pt_rf, pp_rf, out_dir_viz, fname=f"calib_rf_{feature_mode}_{backend.name}.png", title=f"Calibration (RF, {feature_mode}, {backend.name})")

    save_confusion_matrix_heatmap(m_rf.cm, out_dir_viz, fname=f"cm_rf_{feature_mode}_{backend.name}.png", title=f"Confusion matrix (RF, {feature_mode}, {backend.name})", normalize=False)
    save_confusion_matrix_heatmap(m_rf.cm, out_dir_viz, fname=f"cm_rf_{feature_mode}_{backend.name}_norm.png", title=f"Confusion matrix (RF, {feature_mode}, {backend.name})", normalize=True)

    # Persist RF artifacts
    joblib.dump(rf_final, art_dir / f"rf_{feature_mode}_{backend.name}.joblib")
    _save_artifact_json(art_dir / f"rf_threshold_{feature_mode}_{backend.name}.json", {"threshold": float(rf_thr)})

    report["random_forest"] = {
        "best_hparams": {
            "n_estimators": int(best_rf["n_estimators"]),
            "max_features": best_rf["max_features"],
            "max_depth": best_rf["max_depth"],
            "min_samples_split": int(best_rf["min_samples_split"]),
            "min_samples_leaf": int(best_rf["min_samples_leaf"]),
            "max_samples": best_rf["max_samples"],
        },
        "best_cv_f1_illicit": float(best_rf["cv_f1"]),
        "threshold": float(rf_thr),
        "tail_f1_illicit": float(rf_tail_f1),
        "test": m_rf.__dict__,
        "trials_top10": rf_hpo["trials_top10"],
    }

    return report


def main() -> None:
    paths = Paths()
    cfg = ExperimentConfig()
    _ensure_dirs(paths)
    set_seed(cfg.seed, deterministic_torch=False)

    backend = get_tabular_backend()
    rng = np.random.default_rng(cfg.seed)

    write_run_manifest(
        paths.results_dir / "logs",
        phase="04",
        cfg=cfg,
        data_files=[
            paths.processed_dir / "elliptic_labeled.parquet",
            paths.processed_dir / "elliptic_labeled.csv",
        ],
        extra={
            "note": "phase 04 (frozen): backend-aware forward-chaining CV for tabular + tail-step threshold tuning (no test leakage)",
            "tabular_backend": backend.name,
        },
    )

    df = _load_labeled(paths)

    # Phase04: finalize both AF and LF (matches phase03 coverage)
    reports: dict[str, Any] = {
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "tabular_backend": backend.name,
        "modes": {},
    }

    for mode in tqdm([cast(FeatureMode, "AF"), cast(FeatureMode, "LF")], desc="Phase04 modes"):
        rep = _run_mode(df=df, feature_mode=mode, backend=backend, cfg=cfg, paths=paths, rng=rng)
        reports["modes"][str(mode)] = rep

        out_path = paths.results_dir / "metrics" / f"phase04_tuning_{mode}_{backend.name}.json"
        out_path.write_text(json.dumps(_json_safe(rep), indent=2), encoding="utf-8")

    # combined index
    idx_path = paths.results_dir / "metrics" / f"phase04_tuning_index_{backend.name}.json"
    idx_path.write_text(json.dumps(_json_safe(reports), indent=2), encoding="utf-8")

    _persist_artifact_inventory(paths, backend.name)


if __name__ == "__main__":
    main()
