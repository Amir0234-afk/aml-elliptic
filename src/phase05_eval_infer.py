from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data

from src.config import ExperimentConfig, Paths
from src.data import get_feature_cols, load_edges, make_tabular_split
from src.eval import compute_metrics
from src.gnn import GCN, SkipGCN
from src.graph_data import build_graph


def require_tensor(v: Any, name: str) -> Tensor:
    if isinstance(v, torch.Tensor):
        return v
    raise TypeError(f"Expected {name} to be torch.Tensor, got {type(v).__name__}")


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

    df = load_processed(paths)

    out: Dict[str, Any] = {}

    # Canonical temporal test split (AF)
    split_norm = make_tabular_split(df, "AF", train_ratio=cfg.train_ratio, normalize=True)
    split_raw = make_tabular_split(df, "AF", train_ratio=cfg.train_ratio, normalize=False)

    # ---------- Evaluate tuned tabular (if exists) ----------
    tuned_lr_path = paths.results_dir / "model_artifacts" / "tuned_lr_AF.joblib"
    if tuned_lr_path.exists():
        lr = joblib.load(tuned_lr_path)
        yhat = lr.predict(split_norm.X_test)
        out["tuned_logreg_AF"] = compute_metrics(split_norm.y_test, yhat).__dict__

    tuned_rf_path = paths.results_dir / "model_artifacts" / "tuned_rf_AF.joblib"
    if tuned_rf_path.exists():
        rf = joblib.load(tuned_rf_path)
        yhat = rf.predict(split_raw.X_test)
        out["tuned_random_forest_AF"] = compute_metrics(split_raw.y_test, yhat).__dict__

        # Error analysis: FP/FN txIds (first 50)
        y_true = split_raw.y_test
        fp = (yhat == 1) & (y_true == 0)
        fn = (yhat == 0) & (y_true == 1)
        out["rf_error_analysis"] = {
            "false_positive_txIds": split_raw.tx_test[fp].astype(int).tolist()[:50],
            "false_negative_txIds": split_raw.tx_test[fn].astype(int).tolist()[:50],
        }

    # ---------- Evaluate GNN checkpoints (if exist) ----------
    edges_df = load_edges(paths.raw_dir)
    feature_mode = "AF"
    feature_cols = get_feature_cols(df, feature_mode)

    # Same time split used for graph masks
    split_any = make_tabular_split(df, feature_mode, train_ratio=cfg.train_ratio, normalize=False)
    bundle = build_graph(
        df_labeled=df,
        edges_df=edges_df,
        feature_cols=feature_cols,
        train_steps=split_any.train_steps,
        test_steps=split_any.test_steps,
        val_ratio_within_train=cfg.val_ratio_within_train,
        normalize=True,
    )

    data: Data = bundle.data.to(cfg.device)

    def eval_gnn_state(model: torch.nn.Module, state_path: Path) -> dict[str, Any]:
        state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state)
        model = model.to(cfg.device)
        model.eval()

        d = data  # already on device

        x = require_tensor(d.x, "d.x")
        edge_index = require_tensor(d.edge_index, "d.edge_index")
        y = require_tensor(d.y, "d.y").to(torch.long)
        test_mask = require_tensor(d.test_mask, "d.test_mask").to(torch.bool)

        with torch.no_grad():
            logits = model(x, edge_index)
            y_pred = logits.argmax(dim=1)

        y_true_np = y.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        test_idx = test_mask.detach().cpu().numpy().astype(bool)

        return compute_metrics(y_true_np[test_idx], y_pred_np[test_idx]).__dict__

    gcn_path = paths.results_dir / "model_artifacts" / "gcn_AF.pt"
    if gcn_path.exists():
        gcn = GCN(in_dim=len(feature_cols), hidden_dim=cfg.gcn_hidden_dim, dropout=cfg.gcn_dropout)
        out["gcn_AF"] = eval_gnn_state(gcn, gcn_path)

    sgcn_path = paths.results_dir / "model_artifacts" / "skip_gcn_AF.pt"
    if sgcn_path.exists():
        sgcn = SkipGCN(in_dim=len(feature_cols), hidden_dim=cfg.gcn_hidden_dim, dropout=cfg.gcn_dropout)
        out["skip_gcn_AF"] = eval_gnn_state(sgcn, sgcn_path)

    # ---------- Minimal inference demo ----------
    demo: Dict[str, Any] = {}
    sample_tx = int(split_raw.tx_test[0])

    if tuned_rf_path.exists():
        rf = joblib.load(tuned_rf_path)
        row = df.loc[df["txId"] == sample_tx, feature_cols].to_numpy(dtype=np.float32)
        proba = float(rf.predict_proba(row)[0, 1])
        demo["rf_predict_proba_illicit"] = {"txId": sample_tx, "p": proba}

    out["inference_demo"] = demo

    (paths.results_dir / "metrics" / "phase05_final_evaluation.json").write_text(
        json.dumps(out, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
    )


if __name__ == "__main__":
    main()
