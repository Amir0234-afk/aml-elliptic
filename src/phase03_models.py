# src/phase03_models.py
from __future__ import annotations

import csv
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Union, cast

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch_geometric.data import Data

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x

from src.config import ExperimentConfig, Paths
from src.data import FeatureMode, class_weights_binary, get_feature_cols, load_edges
from src.eval import compute_metrics
from src.gnn import GCN, SkipGCN, train_gnn
from src.graph_data import build_graph
from src.repro import set_seed
from src.runlog import write_run_manifest
from src.viz import save_model_comparison_bar


MaxFeat = Union[float, Literal["sqrt", "log2"]]


def require_tensor(v: Any, name: str) -> Tensor:
    if isinstance(v, torch.Tensor):
        return v
    raise TypeError(f"Expected {name} to be torch.Tensor, got {type(v).__name__}")


def ensure_dirs(paths: Paths) -> None:
    (paths.results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "model_artifacts").mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "logs").mkdir(parents=True, exist_ok=True)


def load_processed_labeled(paths: Paths) -> pd.DataFrame:
    parquet_path = paths.processed_dir / "elliptic_labeled.parquet"
    csv_path = paths.processed_dir / "elliptic_labeled.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError("Labeled dataset not found. Run: python -m src.phase01_preprocessing")


def load_processed_full(paths: Paths) -> pd.DataFrame:
    parquet_path = paths.processed_dir / "elliptic_full.parquet"
    csv_path = paths.processed_dir / "elliptic_full.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError("Full dataset not found. Run: python -m src.phase01_preprocessing")


def json_safe(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    return x


def append_experiment_log(paths: Paths, row: Dict[str, Any]) -> None:
    log_path = paths.results_dir / "logs" / "experiments.csv"
    is_new = not log_path.exists()

    fieldnames = [
        "run_id",
        "datetime_utc",
        "phase",
        "model",
        "f1_illicit",
        "precision_illicit",
        "recall_illicit",
    ]

    with log_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in fieldnames})


def _temporal_train_test_steps(df_labeled: pd.DataFrame, train_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    steps = np.sort(df_labeled["time_step"].unique().astype(int))
    n_train = int(np.floor(len(steps) * float(train_ratio)))
    n_train = max(1, min(n_train, len(steps) - 1))
    return steps[:n_train], steps[n_train:]


def _forward_chaining_cv_splits(
    row_steps: np.ndarray,
    train_steps: np.ndarray,
    n_splits: int,
    val_steps: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Leakage-safe folds:
      - train uses early steps
      - val uses immediately following steps
    Returns splits as indices into arrays aligned with `row_steps`.
    """
    train_steps = np.asarray(train_steps, dtype=int)
    steps = np.sort(np.unique(train_steps))
    n = len(steps)
    if n < 2:
        return []

    val_steps = max(1, int(val_steps))
    n_splits = max(1, int(n_splits))

    max_train_end = n - val_steps
    if max_train_end <= 0:
        return []

    min_train_end = max(1, max_train_end // (n_splits + 1))
    ends = np.unique(np.linspace(min_train_end, max_train_end, num=n_splits, dtype=int))

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_end in ends:
        tr_steps = steps[:train_end]
        va_steps = steps[train_end : train_end + val_steps]
        if len(tr_steps) == 0 or len(va_steps) == 0:
            continue

        tr_idx = np.flatnonzero(np.isin(row_steps, tr_steps))
        va_idx = np.flatnonzero(np.isin(row_steps, va_steps))
        if tr_idx.size == 0 or va_idx.size == 0:
            continue
        splits.append((tr_idx, va_idx))

    return splits


def _cv_mean_f1_illicit(
    fit_predict_fn,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    if not splits:
        return -1.0
    scores: list[float] = []
    for tr_idx, va_idx in splits:
        y_pred = fit_predict_fn(X[tr_idx], y[tr_idx], X[va_idx])
        scores.append(float(f1_score(y[va_idx], y_pred, pos_label=1)))
    return float(np.mean(scores)) if scores else -1.0


def _make_logreg(
    C: float,
    cw: dict[int, float],
    seed: int,
    max_iter: int,
    tol: float,
) -> LogisticRegression:
    """
    scikit-learn >=1.8: passing `penalty=` emits a FutureWarning.
    Their warning explicitly suggests using l1_ratio instead of penalty.

    Strategy:
      1) Try "new style": no penalty arg, set l1_ratio=0.0 (L2) with solver='saga'
      2) If that fails (older sklearn semantics), fall back to penalty='l2'
    """
    try:
        return LogisticRegression(
            C=float(C),
            solver="saga",
            l1_ratio=0.0,  # L2 per sklearn >= 1.8 FutureWarning guidance
            max_iter=int(max_iter),
            tol=float(tol),
            class_weight=cw,
            random_state=int(seed),
        )
    except Exception:
        # Older sklearn: l1_ratio requires penalty='elasticnet'; use classic L2.
        return LogisticRegression(
            C=float(C),
            solver="saga",
            penalty="l2",
            max_iter=int(max_iter),
            tol=float(tol),
            class_weight=cw,
            random_state=int(seed),
        )


def run_tabular_models(
    df_labeled: pd.DataFrame,
    feature_mode: FeatureMode,
    cfg: ExperimentConfig,
    paths: Paths,
) -> dict:
    """
    Tabular:
      - temporal split by time_step
      - forward-chaining CV inside train for HPO (no leakage)
    NOTE: scikit-learn tabular models run on CPU (no CUDA).
    """
    df = df_labeled.copy()
    df = df.sort_values(["time_step", "txId"]).reset_index(drop=True)

    train_steps, test_steps = _temporal_train_test_steps(df, cfg.train_ratio)
    feature_cols = get_feature_cols(df, feature_mode)

    df_train = df[df["time_step"].isin(train_steps)].copy()
    df_test = df[df["time_step"].isin(test_steps)].copy()

    X_train_raw = df_train[feature_cols].to_numpy(dtype=np.float32)
    X_test_raw = df_test[feature_cols].to_numpy(dtype=np.float32)

    # defensive NaN handling
    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_raw = np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)

    y_train = df_train["class"].to_numpy(dtype=np.int64)
    y_test = df_test["class"].to_numpy(dtype=np.int64)

    cw = class_weights_binary(y_train)

    row_steps = df_train["time_step"].to_numpy(dtype=int)
    cv_splits = _forward_chaining_cv_splits(
        row_steps=row_steps,
        train_steps=train_steps,
        n_splits=int(cfg.tabular_cv_splits),
        val_steps=int(cfg.tabular_cv_val_steps),
    )

    # ---------------- LR (normalized) ----------------
    def fit_predict_lr(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, C_: float) -> np.ndarray:
        scaler = StandardScaler()
        Xtr_n = scaler.fit_transform(Xtr)
        Xva_n = scaler.transform(Xva)

        lr = _make_logreg(
            C=float(C_),
            cw=cw,
            seed=cfg.seed,
            max_iter=cfg.lr_max_iter,
            tol=cfg.lr_tol,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            lr.fit(Xtr_n, ytr)

        return lr.predict(Xva_n)

    best_lr_C = 1.0
    if bool(cfg.tabular_tune) and cv_splits:
        C_grid = np.logspace(-3, 2, num=20).astype(float)
        rng = np.random.default_rng(cfg.seed)
        trials = min(int(cfg.tabular_tune_trials), len(C_grid))
        cand_C = rng.choice(C_grid, size=trials, replace=False)

        best_score = -1.0
        for C_ in tqdm(cand_C, desc=f"LR HPO ({feature_mode})", leave=False):
            score = _cv_mean_f1_illicit(
                lambda Xtr, ytr, Xva: fit_predict_lr(Xtr, ytr, Xva, C_=float(C_)),
                X_train_raw,
                y_train,
                cv_splits,
            )
            if score > best_score:
                best_score = score
                best_lr_C = float(C_)

    lr_scaler = StandardScaler()
    X_train_lr = lr_scaler.fit_transform(X_train_raw)
    X_test_lr = lr_scaler.transform(X_test_raw)

    lr_final = _make_logreg(
        C=float(best_lr_C),
        cw=cw,
        seed=cfg.seed,
        max_iter=cfg.lr_max_iter,
        tol=cfg.lr_tol,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        lr_final.fit(X_train_lr, y_train)

    yhat_lr = lr_final.predict(X_test_lr)
    m_lr = compute_metrics(y_test, yhat_lr)

    # ---------------- RF (raw) ----------------
    def fit_predict_rf(
        Xtr: np.ndarray,
        ytr: np.ndarray,
        Xva: np.ndarray,
        n_estimators: int,
        max_features: MaxFeat,
        max_depth: int | None,
        min_samples_leaf: int,
    ) -> np.ndarray:
        rf = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_features=max_features,
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            class_weight=cw,
            n_jobs=-1,
            random_state=cfg.seed,
        )
        rf.fit(Xtr, ytr)
        return rf.predict(Xva)

    best_rf_params: dict[str, Any] = {
        "n_estimators": int(cfg.rf_n_estimators),
        "max_features": cast(MaxFeat, cfg.rf_max_features),
        "max_depth": None,
        "min_samples_leaf": 1,
    }

    if bool(cfg.tabular_tune) and cv_splits:
        rng = np.random.default_rng(cfg.seed)

        n_estimators_grid = np.array([200, 400, 600, 800, 1000], dtype=int)
        max_features_grid: list[MaxFeat] = ["sqrt", 0.3, 0.5, 0.7]
        max_depth_grid: list[int | None] = [None, 10, 20, 30, 40]
        min_leaf_grid = np.array([1, 2, 5, 10], dtype=int)

        best_score = -1.0
        for _ in tqdm(range(int(cfg.tabular_tune_trials)), desc=f"RF HPO ({feature_mode})", leave=False):
            mf = max_features_grid[int(rng.integers(0, len(max_features_grid)))]
            md = max_depth_grid[int(rng.integers(0, len(max_depth_grid)))]
            params = {
                "n_estimators": int(rng.choice(n_estimators_grid)),
                "max_features": mf,
                "max_depth": md,
                "min_samples_leaf": int(rng.choice(min_leaf_grid)),
            }

            score = _cv_mean_f1_illicit(
                lambda Xtr, ytr, Xva: fit_predict_rf(
                    Xtr,
                    ytr,
                    Xva,
                    n_estimators=params["n_estimators"],
                    max_features=cast(MaxFeat, params["max_features"]),
                    max_depth=cast(int | None, params["max_depth"]),
                    min_samples_leaf=params["min_samples_leaf"],
                ),
                X_train_raw,
                y_train,
                cv_splits,
            )
            if score > best_score:
                best_score = score
                best_rf_params = params

    rf_final = RandomForestClassifier(
        n_estimators=int(best_rf_params["n_estimators"]),
        max_features=cast(MaxFeat, best_rf_params["max_features"]),
        max_depth=cast(int | None, best_rf_params["max_depth"]),
        min_samples_leaf=int(best_rf_params["min_samples_leaf"]),
        class_weight=cw,
        n_jobs=-1,
        random_state=cfg.seed,
    )
    rf_final.fit(X_train_raw, y_train)
    yhat_rf = rf_final.predict(X_test_raw)
    m_rf = compute_metrics(y_test, yhat_rf)

    results = {
        "feature_mode": feature_mode,
        "train_steps": train_steps.tolist(),
        "test_steps": test_steps.tolist(),
        "class_weights": {str(k): float(v) for k, v in cw.items()},
        "tuning": {
            "enabled": bool(cfg.tabular_tune),
            "cv_splits": int(len(cv_splits)),
            "lr_best_C": float(best_lr_C),
            "rf_best_params": best_rf_params,
        },
        "logreg": m_lr.__dict__,
        "random_forest": m_rf.__dict__,
    }

    # Persist TABULAR baselines
    (paths.results_dir / "metrics" / f"baselines_{feature_mode}.json").write_text(
        json.dumps(json_safe(results), indent=2),
        encoding="utf-8",
    )

    base = paths.results_dir / "model_artifacts"
    joblib.dump(lr_final, base / f"lr_{feature_mode}.joblib")
    joblib.dump(lr_scaler, base / f"scaler_{feature_mode}.joblib")
    joblib.dump(rf_final, base / f"rf_{feature_mode}.joblib")

    rid = f"tabular_{feature_mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    append_experiment_log(
        paths,
        {
            "run_id": rid,
            "datetime_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "03",
            "model": f"LR_{feature_mode}",
            "f1_illicit": m_lr.f1_illicit,
            "precision_illicit": m_lr.precision_illicit,
            "recall_illicit": m_lr.recall_illicit,
        },
    )
    append_experiment_log(
        paths,
        {
            "run_id": rid,
            "datetime_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "03",
            "model": f"RF_{feature_mode}",
            "f1_illicit": m_rf.f1_illicit,
            "precision_illicit": m_rf.precision_illicit,
            "recall_illicit": m_rf.recall_illicit,
        },
    )

    return results


def _node_embeddings(model: torch.nn.Module, x: Tensor, edge_index: Tensor) -> Tensor:
    enc = getattr(model, "encode", None)
    if callable(enc):
        out = enc(x, edge_index)
        if not isinstance(out, torch.Tensor):
            raise TypeError("model.encode(...) did not return a torch.Tensor")
        return out
    raise AttributeError("Model has no .encode(x, edge_index). Add encode() to your GCN/SkipGCN to enable embedding_aug.")


def run_gnn_models(
    df_full: pd.DataFrame,     # FULL DATASET (includes class=-1)
    df_labeled: pd.DataFrame,  # LABELED ONLY (for split steps)
    edges_df: pd.DataFrame,
    feature_mode: FeatureMode,
    cfg: ExperimentConfig,
    paths: Paths,
) -> dict:
    device = cfg.device

    train_steps, test_steps = _temporal_train_test_steps(df_labeled, cfg.train_ratio)
    feature_cols = get_feature_cols(df_full, feature_mode)

    bundle = build_graph(
        df_labeled=df_full,
        edges_df=edges_df,
        feature_cols=feature_cols,
        train_steps=train_steps,
        test_steps=test_steps,
        val_ratio_within_train=float(cfg.val_ratio_within_train),
        normalize=True,
        make_undirected=True,
    )
    data: Data = bundle.data

    x = require_tensor(data.x, "data.x")
    edge_index = require_tensor(data.edge_index, "data.edge_index")
    y = require_tensor(data.y, "data.y").to(torch.long)
    train_mask = require_tensor(data.train_mask, "data.train_mask").to(torch.bool)
    val_mask = require_tensor(data.val_mask, "data.val_mask").to(torch.bool)
    test_mask = require_tensor(data.test_mask, "data.test_mask").to(torch.bool)

    if val_mask.sum().item() == 0:
        val_mask = train_mask.clone()

    y_train_np = y[train_mask].detach().cpu().numpy().astype(int)
    cw = class_weights_binary(y_train_np)
    base_weights: tuple[float, float] = (float(cw[0]), float(cw[1]))

    out: Dict[str, Any] = {
        "feature_mode": feature_mode,
        "train_steps": bundle.train_steps.tolist(),
        "val_steps": bundle.val_steps.tolist(),
        "test_steps": bundle.test_steps.tolist(),
        "device": device,
        "class_weights_train": {str(k): float(v) for k, v in cw.items()},
        "graph": {
            "num_nodes": int(x.size(0)),
            "num_edges": int(edge_index.size(1)),
            "train_nodes_labeled": int(train_mask.sum().item()),
            "val_nodes_labeled": int(val_mask.sum().item()),
            "test_nodes_labeled": int(test_mask.sum().item()),
            "unlabeled_nodes": int((y == -1).sum().item()),
        },
    }

    def eval_state(model: torch.nn.Module, state: dict[str, Tensor]) -> dict[str, Any]:
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()

        d = data.to(device)
        x_local = require_tensor(d.x, "d.x")
        ei_local = require_tensor(d.edge_index, "d.edge_index")
        y_local = require_tensor(d.y, "d.y").to(torch.long)
        test_mask_local = require_tensor(d.test_mask, "d.test_mask").to(torch.bool)

        with torch.no_grad():
            logits = model(x_local, ei_local)
            y_pred = logits.argmax(dim=1)

        y_true_np = y_local.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        test_idx = test_mask_local.detach().cpu().numpy().astype(bool)

        m = compute_metrics(y_true_np[test_idx], y_pred_np[test_idx])
        return m.__dict__

    rng = np.random.default_rng(cfg.seed)
    weight_scales: list[float] = [1.0]
    if bool(cfg.gnn_tune_class_weight):
        weight_scales = [0.5, 1.0, 2.0, 4.0]

    def train_one(
        model_ctor,
        hidden_dim: int,
        dropout: float,
        lr: float,
        wd: float,
        wscale: float,
    ) -> tuple[float, dict[str, Tensor], int, dict[str, Any]]:
        model = model_ctor(in_dim=len(feature_cols), hidden_dim=hidden_dim, dropout=dropout)
        w: tuple[float, float] = (base_weights[0], base_weights[1] * float(wscale))

        res = train_gnn(
            model,
            data,
            train_mask=train_mask,
            monitor_mask=val_mask,
            class_weights=w,
            lr=float(lr),
            weight_decay=float(wd),
            epochs=int(cfg.epochs),
            patience=int(cfg.patience),
            device=device,
            show_progress=True,
        )
        hp = {
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "lr": float(lr),
            "weight_decay": float(wd),
            "illicit_wscale": float(wscale),
        }
        return float(res.best_monitor_f1), res.best_state_dict, int(res.best_epoch), hp

    def tune_arch(model_name: str, model_ctor) -> dict[str, Any]:
        trials = max(1, int(cfg.gnn_tune_trials))
        best_f1 = -1.0
        best_state: dict[str, Tensor] | None = None
        best_epoch = 0
        best_hp: dict[str, Any] | None = None

        hidden_grid = [50, 100, 150, 200]
        drop_grid = [0.0, 0.2, 0.5]
        lr_grid = [5e-4, 1e-3, 2e-3, 5e-3]
        wd_grid = [0.0, 1e-6, 1e-5, 1e-4, 5e-4]

        for _ in tqdm(range(trials), desc=f"{model_name} HPO ({feature_mode}, {device})", leave=False):
            hidden = int(hidden_grid[int(rng.integers(0, len(hidden_grid)))])
            drop = float(drop_grid[int(rng.integers(0, len(drop_grid)))])
            lr_ = float(lr_grid[int(rng.integers(0, len(lr_grid)))])
            wd_ = float(wd_grid[int(rng.integers(0, len(wd_grid)))])
            wscale = float(weight_scales[int(rng.integers(0, len(weight_scales)))])

            f1v, state, ep, hp = train_one(model_ctor, hidden, drop, lr_, wd_, wscale)
            if f1v > best_f1:
                best_f1 = f1v
                best_state = state
                best_epoch = ep
                best_hp = hp

        if best_state is None or best_hp is None:
            raise RuntimeError(f"{model_name} tuning produced no valid state (trials={trials}).")

        return {"f1": float(best_f1), "state": best_state, "epoch": int(best_epoch), "hp": best_hp}

    best_gcn = tune_arch("GCN", GCN)
    out["gcn"] = {
        "best_val_f1": best_gcn["f1"],
        "best_epoch": best_gcn["epoch"],
        "best_hp": best_gcn["hp"],
        "test_metrics": eval_state(
            GCN(
                in_dim=len(feature_cols),
                hidden_dim=int(best_gcn["hp"]["hidden_dim"]),
                dropout=float(best_gcn["hp"]["dropout"]),
            ),
            cast(dict[str, Tensor], best_gcn["state"]),
        ),
    }

    best_sgcn = tune_arch("SkipGCN", SkipGCN)
    out["skip_gcn"] = {
        "best_val_f1": best_sgcn["f1"],
        "best_epoch": best_sgcn["epoch"],
        "best_hp": best_sgcn["hp"],
        "test_metrics": eval_state(
            SkipGCN(
                in_dim=len(feature_cols),
                hidden_dim=int(best_sgcn["hp"]["hidden_dim"]),
                dropout=float(best_sgcn["hp"]["dropout"]),
            ),
            cast(dict[str, Tensor], best_sgcn["state"]),
        ),
    }

    # --- Embedding augmentation: RF on (x ⊕ embedding) ---
    if bool(cfg.enable_embedding_aug):
        arch = str(cfg.embedding_aug_model).lower().strip()
        if arch == "skip_gcn":
            model_ctor = SkipGCN
            state = cast(dict[str, Tensor], best_sgcn["state"])
            hp = cast(dict[str, Any], best_sgcn["hp"])
        else:
            arch = "gcn"
            model_ctor = GCN
            state = cast(dict[str, Tensor], best_gcn["state"])
            hp = cast(dict[str, Any], best_gcn["hp"])

        emb_model = model_ctor(
            in_dim=len(feature_cols),
            hidden_dim=int(hp["hidden_dim"]),
            dropout=float(hp["dropout"]),
        )
        emb_model.load_state_dict(state)
        emb_model = emb_model.to(device)
        emb_model.eval()

        d = data.to(device)
        x_t = require_tensor(d.x, "d.x")
        ei_t = require_tensor(d.edge_index, "d.edge_index")
        y_t = require_tensor(d.y, "d.y")

        with torch.no_grad():
            h = _node_embeddings(emb_model, x_t, ei_t)

        h_np = h.detach().cpu().numpy().astype(np.float32)
        x_np = x_t.detach().cpu().numpy().astype(np.float32)
        X_aug = np.concatenate([x_np, h_np], axis=1)

        y_np = y_t.detach().cpu().numpy().astype(int)

        ts_np = df_full["time_step"].to_numpy(dtype=int)
        labeled = (y_np != -1)

        train_idx = labeled & np.isin(ts_np, train_steps)
        test_idx = labeled & np.isin(ts_np, test_steps)

        y_tr = y_np[train_idx]
        cw2 = class_weights_binary(y_tr)

        rf_aug = RandomForestClassifier(
            n_estimators=int(cfg.embedding_aug_rf_n_estimators),
            max_features="sqrt",
            class_weight=cw2,
            n_jobs=-1,
            random_state=cfg.seed,
        )
        rf_aug.fit(X_aug[train_idx], y_tr)
        y_pred = rf_aug.predict(X_aug[test_idx])
        m_aug = compute_metrics(y_np[test_idx], y_pred)

        out["rf_on_gnn_embeddings"] = {
            "arch": arch,
            "n_estimators": int(cfg.embedding_aug_rf_n_estimators),
            "test_metrics": m_aug.__dict__,
        }

        base = paths.results_dir / "model_artifacts"
        joblib.dump(rf_aug, base / f"rf_aug_{feature_mode}_{arch}.joblib")

    # Persist GNN metrics (separate file; do not overwrite baselines_*.json)
    (paths.results_dir / "metrics" / f"gnn_{feature_mode}.json").write_text(
        json.dumps(json_safe(out), indent=2),
        encoding="utf-8",
    )

    base = paths.results_dir / "model_artifacts"
    torch.save(cast(dict[str, Tensor], best_gcn["state"]), base / f"gcn_{feature_mode}.pt")
    torch.save(cast(dict[str, Tensor], best_sgcn["state"]), base / f"skip_gcn_{feature_mode}.pt")

    rid = f"gnn_{feature_mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    append_experiment_log(
        paths,
        {
            "run_id": rid,
            "datetime_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "03",
            "model": f"GCN_{feature_mode}",
            "f1_illicit": out["gcn"]["test_metrics"]["f1_illicit"],
            "precision_illicit": out["gcn"]["test_metrics"]["precision_illicit"],
            "recall_illicit": out["gcn"]["test_metrics"]["recall_illicit"],
        },
    )
    append_experiment_log(
        paths,
        {
            "run_id": rid,
            "datetime_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "03",
            "model": f"SkipGCN_{feature_mode}",
            "f1_illicit": out["skip_gcn"]["test_metrics"]["f1_illicit"],
            "precision_illicit": out["skip_gcn"]["test_metrics"]["precision_illicit"],
            "recall_illicit": out["skip_gcn"]["test_metrics"]["recall_illicit"],
        },
    )

    if "rf_on_gnn_embeddings" in out:
        append_experiment_log(
            paths,
            {
                "run_id": rid,
                "datetime_utc": datetime.now(timezone.utc).isoformat(),
                "phase": "03",
                "model": f"RF_AUG_{feature_mode}_{out['rf_on_gnn_embeddings']['arch']}",
                "f1_illicit": out["rf_on_gnn_embeddings"]["test_metrics"]["f1_illicit"],
                "precision_illicit": out["rf_on_gnn_embeddings"]["test_metrics"]["precision_illicit"],
                "recall_illicit": out["rf_on_gnn_embeddings"]["test_metrics"]["recall_illicit"],
            },
        )

    return out


def main() -> None:
    paths = Paths()
    cfg = ExperimentConfig()
    ensure_dirs(paths)
    set_seed(cfg.seed, deterministic_torch=False)

    write_run_manifest(
        paths.results_dir / "logs",
        phase="03",
        cfg=cfg,
        data_files=[
            paths.raw_dir / "elliptic_txs_features.csv",
            paths.raw_dir / "elliptic_txs_edgelist.csv",
            paths.raw_dir / "elliptic_txs_classes.csv",
            paths.processed_dir / "elliptic_full.csv",
            paths.processed_dir / "elliptic_labeled.csv",
        ],
        extra={"note": f"phase 03: temporal CV tuning for tabular; tuned GNN + optional embedding augmentation; device={cfg.device}"},
    )

    df_labeled = load_processed_labeled(paths)
    df_full = load_processed_full(paths)
    edges_df = load_edges(paths.raw_dir)

    res_af = run_tabular_models(df_labeled, cast(FeatureMode, "AF"), cfg, paths)
    _res_lf = run_tabular_models(df_labeled, cast(FeatureMode, "LF"), cfg, paths)

    save_model_comparison_bar(
        {"LR(AF)": res_af["logreg"]["f1_illicit"], "RF(AF)": res_af["random_forest"]["f1_illicit"]},
        paths.results_dir / "visualizations",
    )

    if bool(cfg.enable_gnn):
        for mode in tqdm(list(cfg.gnn_feature_modes), desc="GNN feature modes"):
            fm = cast(FeatureMode, mode)
            run_gnn_models(df_full, df_labeled, edges_df, fm, cfg, paths)


if __name__ == "__main__":
    main()
