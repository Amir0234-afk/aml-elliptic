from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data

from src.baselines import predict_proba_positive, train_lr, train_rf
from src.config import ExperimentConfig, Paths
from src.data import (
    class_weights_binary,
    get_feature_cols,
    load_edges,
    make_tabular_split,
)
from src.eval import compute_metrics, threshold_predictions
from src.gnn import GCN, SkipGCN, train_gnn
from src.graph_data import build_graph
from src.viz import save_model_comparison_bar
from src.repro import set_seed
from src.runlog import write_run_manifest

def require_tensor(v: Any, name: str) -> Tensor:
    """Runtime + type-narrowing guard for PyG Data optional/union-typed attributes."""
    if isinstance(v, torch.Tensor):
        return v
    raise TypeError(f"Expected {name} to be torch.Tensor, got {type(v).__name__}")



def ensure_dirs(paths: Paths) -> None:
    (paths.results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "model_artifacts").mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "logs").mkdir(parents=True, exist_ok=True)


def load_processed(paths: Paths) -> pd.DataFrame:
    parquet_path = paths.processed_dir / "elliptic_labeled.parquet"
    csv_path = paths.processed_dir / "elliptic_labeled.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError("Processed dataset not found. Run: python -m src.phase01_preprocessing")


def append_experiment_log(paths: Paths, row: Dict[str, Any]) -> None:
    log_path = paths.results_dir / "logs" / "experiments.csv"
    is_new = not log_path.exists()

    # Fixed schema to avoid CSV header mismatch across runs
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


def run_tabular_baselines(
    df_labeled: pd.DataFrame,
    feature_mode: str,
    cfg: ExperimentConfig,
    paths: Paths,
) -> dict:
    split_norm = make_tabular_split(
        df_labeled, feature_mode=feature_mode, train_ratio=cfg.train_ratio, normalize=True
    )
    split_raw = make_tabular_split(
        df_labeled, feature_mode=feature_mode, train_ratio=cfg.train_ratio, normalize=False
    )

    cw = class_weights_binary(split_norm.y_train)

    # LR (normalized)
    lr = train_lr(split_norm.X_train, split_norm.y_train, class_weight=cw)
    p_lr = predict_proba_positive(lr, split_norm.X_test)
    yhat_lr = threshold_predictions(p_lr, 0.5)
    m_lr = compute_metrics(split_norm.y_test, yhat_lr)

    # RF (raw)
    rf = train_rf(
        split_raw.X_train,
        split_raw.y_train,
        cfg.rf_n_estimators,
        cfg.rf_max_features,
        class_weight=cw,
    )
    p_rf = predict_proba_positive(rf, split_raw.X_test)
    yhat_rf = threshold_predictions(p_rf, 0.5)
    m_rf = compute_metrics(split_raw.y_test, yhat_rf)

    results = {
        "feature_mode": feature_mode,
        "train_steps": split_norm.train_steps.tolist(),
        "test_steps": split_norm.test_steps.tolist(),
        "class_weights": cw,
        "logreg": m_lr.__dict__,
        "random_forest": m_rf.__dict__,
    }

    # Save metrics
    (paths.results_dir / "metrics" / f"baselines_{feature_mode}.json").write_text(
        json.dumps(results, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
    )

    # Save artifacts
    base = paths.results_dir / "model_artifacts"
    joblib.dump(lr, base / f"lr_{feature_mode}.joblib")
    joblib.dump(rf, base / f"rf_{feature_mode}.joblib")
    if split_norm.scaler is not None:
        joblib.dump(split_norm.scaler, base / f"scaler_{feature_mode}.joblib")

    meta = {
        "feature_mode": feature_mode,
        "feature_cols": split_norm.feature_cols,
        "train_steps": split_norm.train_steps.tolist(),
        "test_steps": split_norm.test_steps.tolist(),
        "normalize_lr": True,
        "normalize_rf": False,
    }
    (base / f"tabular_meta_{feature_mode}.json").write_text(json.dumps(meta, indent=2))

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


def run_gnn_models(
    df_labeled: pd.DataFrame,
    edges_df: pd.DataFrame,
    feature_mode: str,
    cfg: ExperimentConfig,
    paths: Paths,
) -> dict:
    # Use the same temporal split as tabular
    split_any = make_tabular_split(df_labeled, feature_mode=feature_mode, train_ratio=cfg.train_ratio, normalize=False)
    train_steps = split_any.train_steps
    test_steps = split_any.test_steps

    feature_cols = get_feature_cols(df_labeled, feature_mode)

    bundle = build_graph(
        df_labeled=df_labeled,
        edges_df=edges_df,
        feature_cols=feature_cols,
        train_steps=train_steps,
        test_steps=test_steps,
        val_ratio_within_train=cfg.val_ratio_within_train,
        normalize=True,
    )
    data: Data = bundle.data

    # ---- Type narrowing (fixes Pylance) ----
    x = require_tensor(data.x, "data.x")
    edge_index = require_tensor(data.edge_index, "data.edge_index")
    y = require_tensor(data.y, "data.y").to(torch.long)
    train_mask = require_tensor(data.train_mask, "data.train_mask").to(torch.bool)
    val_mask = require_tensor(data.val_mask, "data.val_mask").to(torch.bool)
    test_mask = require_tensor(data.test_mask, "data.test_mask").to(torch.bool)

    # class weights from TRAIN NODES
    y_train_np = y[train_mask].detach().cpu().numpy()
    cw = class_weights_binary(y_train_np)
    class_weights = (cw[0], cw[1])

    out: Dict[str, Any] = {
        "feature_mode": feature_mode,
        "train_steps": train_steps.tolist(),
        "test_steps": test_steps.tolist(),
        "class_weights": {str(k): v for k, v in cw.items()},
        "graph": {
            "num_nodes": int(x.size(0)),
            "num_edges": int(edge_index.size(1)),
            "train_nodes": int(train_mask.sum().item()),
            "val_nodes": int(val_mask.sum().item()),
            "test_nodes": int(test_mask.sum().item()),
        },
    }

    def eval_model(model: torch.nn.Module, state: dict) -> dict[str, Any]:
        model.load_state_dict(state)
        model = model.to(cfg.device)
        model.eval()

        d = data.to(cfg.device)
        y_local = require_tensor(d.y, "d.y").to(torch.long)
        test_mask_local = require_tensor(d.test_mask, "d.test_mask").to(torch.bool)

        with torch.no_grad():
            logits = model(require_tensor(d.x, "d.x"), require_tensor(d.edge_index, "d.edge_index"))
            y_pred = logits.argmax(dim=1)

        y_true_np = y_local.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        test_idx = test_mask_local.detach().cpu().numpy().astype(bool)

        m = compute_metrics(y_true_np[test_idx], y_pred_np[test_idx])
        return m.__dict__

    # GCN
    gcn = GCN(in_dim=len(feature_cols), hidden_dim=cfg.gcn_hidden_dim, dropout=cfg.gcn_dropout)
    res_gcn = train_gnn(
        gcn,
        data,
        train_mask,
        val_mask,
        class_weights=class_weights,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        epochs=cfg.epochs,
        patience=cfg.patience,
        device=cfg.device,
    )
    out["gcn"] = {
        "best_val_f1": res_gcn.best_val_f1,
        "test_metrics": eval_model(gcn, res_gcn.best_state_dict),
    }

    # Skip-GCN
    sgcn = SkipGCN(in_dim=len(feature_cols), hidden_dim=cfg.gcn_hidden_dim, dropout=cfg.gcn_dropout)
    res_sgcn = train_gnn(
        sgcn,
        data,
        train_mask,
        val_mask,
        class_weights=class_weights,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        epochs=cfg.epochs,
        patience=cfg.patience,
        device=cfg.device,
    )
    out["skip_gcn"] = {
        "best_val_f1": res_sgcn.best_val_f1,
        "test_metrics": eval_model(sgcn, res_sgcn.best_state_dict),
    }

    # Save metrics + artifacts
    (paths.results_dir / "metrics" / f"gnn_{feature_mode}.json").write_text(
        json.dumps(out, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
    )
    base = paths.results_dir / "model_artifacts"
    torch.save(res_gcn.best_state_dict, base / f"gcn_{feature_mode}.pt")
    torch.save(res_sgcn.best_state_dict, base / f"skip_gcn_{feature_mode}.pt")

    meta = {
        "feature_mode": feature_mode,
        "feature_cols": feature_cols,
        "train_steps": train_steps.tolist(),
        "test_steps": test_steps.tolist(),
        "val_ratio_within_train": cfg.val_ratio_within_train,
        "normalized": True,
    }
    (base / f"gnn_meta_{feature_mode}.json").write_text(json.dumps(meta, indent=2))

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
        data_files=[paths.raw_dir / "elliptic_txs_features.csv",paths.raw_dir / "elliptic_txs_edgelist.csv",paths.raw_dir / "elliptic_txs_classes.csv",paths.processed_dir / "elliptic_labeled.parquet",paths.processed_dir / "elliptic_labeled.csv",],
        extra={"note": "phase 03 baselines and GNN training"},
    )

    df_labeled = load_processed(paths)
    edges_df = load_edges(paths.raw_dir)

    # Tabular baselines
    res_af = run_tabular_baselines(df_labeled, "AF", cfg, paths)
    _res_lf = run_tabular_baselines(df_labeled, "LF", cfg, paths)

    # Comparison plot (baseline)
    save_model_comparison_bar(
        {"LR(AF)": res_af["logreg"]["f1_illicit"], "RF(AF)": res_af["random_forest"]["f1_illicit"]},
        paths.results_dir / "visualizations",
    )

    # GNN models
    if cfg.enable_gnn:
        for mode in cfg.gnn_feature_modes:
            run_gnn_models(df_labeled, edges_df, mode, cfg, paths)


if __name__ == "__main__":
    main()
