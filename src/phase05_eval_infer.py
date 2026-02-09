# src/phase05_eval_infer.py
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, cast

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data

from src.config import ExperimentConfig, Paths
from src.data import FeatureMode, get_feature_cols, load_edges
from src.eval import compute_metrics
from src.gnn import GCN, SkipGCN
from src.graph_data import build_graph
from src.repro import set_seed
from src.runlog import write_run_manifest


def _read_json(path: Path) -> Dict[str, Any]:
    return cast(Dict[str, Any], json.loads(path.read_text(encoding="utf-8")))



def json_safe(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x



def _load_processed_full(paths: Paths) -> pd.DataFrame:
    p = paths.processed_dir / "elliptic_full.parquet"
    c = paths.processed_dir / "elliptic_full.csv"
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError("Full dataset not found. Run: python -m src.phase01_preprocessing")


def _load_processed_labeled(paths: Paths) -> pd.DataFrame:
    p = paths.processed_dir / "elliptic_labeled.parquet"
    c = paths.processed_dir / "elliptic_labeled.csv"
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError("Labeled dataset not found. Run: python -m src.phase01_preprocessing")


def _require_tensor(v: Any, name: str) -> Tensor:
    if isinstance(v, torch.Tensor):
        return v
    raise TypeError(f"Expected {name} to be torch.Tensor, got {type(v).__name__}")


def _pick_device(cfg: ExperimentConfig) -> torch.device:
    dev = str(getattr(cfg, "device", "cpu")).strip().lower()
    if dev.startswith("cuda") and torch.cuda.is_available():
        # supports "cuda" or "cuda:0"
        return torch.device(dev)
    return torch.device("cpu")


def _temporal_train_test_steps(df_labeled: pd.DataFrame, train_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    steps = np.sort(df_labeled["time_step"].unique().astype(int))
    n_train = int(np.floor(len(steps) * float(train_ratio)))
    n_train = max(1, min(n_train, len(steps) - 1))
    return steps[:n_train], steps[n_train:]


def _build_graph_for_mode(
    *,
    df_full: pd.DataFrame,
    df_labeled: pd.DataFrame,
    edges_df: pd.DataFrame,
    feature_mode: FeatureMode,
    cfg: ExperimentConfig,
) -> tuple[Data, np.ndarray, np.ndarray]:
    """
    Builds the exact same kind of graph bundle as phase03 (normalization, undirected, val split within train).
    Returns (data, train_steps, test_steps).
    """
    train_steps, test_steps = _temporal_train_test_steps(df_labeled, cfg.train_ratio)
    feature_cols = get_feature_cols(df_full, feature_mode)

    bundle = build_graph(
        df_labeled=df_full,  # NOTE: in your phase03 you pass df_full here
        edges_df=edges_df,
        feature_cols=feature_cols,
        train_steps=train_steps,
        test_steps=test_steps,
        val_ratio_within_train=float(cfg.val_ratio_within_train),
        normalize=True,
        make_undirected=True,
    )

    data: Data = bundle.data
    return data, train_steps, test_steps


def _eval_loaded_model(model: torch.nn.Module, data: Data, device: torch.device) -> dict[str, Any]:
    model = model.to(device)
    model.eval()

    d = data.to(device)
    x = _require_tensor(d.x, "data.x")
    ei = _require_tensor(d.edge_index, "data.edge_index")
    y = _require_tensor(d.y, "data.y").to(torch.long)
    test_mask = _require_tensor(d.test_mask, "data.test_mask").to(torch.bool)

    with torch.no_grad():
        logits = model(x, ei)
        y_pred = logits.argmax(dim=1)

    y_true_np = y.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    test_idx = test_mask.detach().cpu().numpy().astype(bool)

    m = compute_metrics(y_true_np[test_idx], y_pred_np[test_idx])
    return m.__dict__


def _load_hp_from_gnn_metrics(paths: Paths, feature_mode: FeatureMode) -> dict[str, Any]:
    """
    Reads results/metrics/gnn_{mode}.json and returns:
      {"gcn_hp": {...}, "skip_gcn_hp": {...}}
    """
    p = paths.results_dir / "metrics" / f"gnn_{feature_mode}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run phase03 first.")
    info = _read_json(p)
    return {
        "gcn_hp": cast(dict[str, Any], info["gcn"]["best_hp"]),
        "skip_gcn_hp": cast(dict[str, Any], info["skip_gcn"]["best_hp"]),
    }


def _instantiate_gcn(in_dim: int, hp: dict[str, Any]) -> GCN:
    return GCN(
        in_dim=int(in_dim),
        hidden_dim=int(hp["hidden_dim"]),
        dropout=float(hp.get("dropout", 0.0)),
    )


def _instantiate_skipgcn(in_dim: int, hp: dict[str, Any]) -> SkipGCN:
    return SkipGCN(
        in_dim=int(in_dim),
        hidden_dim=int(hp["hidden_dim"]),
        dropout=float(hp.get("dropout", 0.0)),
    )


def main() -> None:
    paths = Paths()
    cfg = ExperimentConfig()
    set_seed(cfg.seed, deterministic_torch=False)

    (paths.results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "logs").mkdir(parents=True, exist_ok=True)

    write_run_manifest(
        paths.results_dir / "logs",
        phase="05",
        cfg=cfg,
        data_files=[
            paths.processed_dir / "elliptic_full.parquet",
            paths.processed_dir / "elliptic_full.csv",
            paths.processed_dir / "elliptic_labeled.parquet",
            paths.processed_dir / "elliptic_labeled.csv",
            paths.raw_dir / "elliptic_txs_edgelist.csv",
        ],
        extra={"note": "phase 05: evaluate saved checkpoints using hp from gnn_*.json to avoid shape mismatch"},
    )

    device = _pick_device(cfg)

    # Load data
    df_full = _load_processed_full(paths)
    df_labeled = _load_processed_labeled(paths)
    edges_df = load_edges(paths.raw_dir)

    out: dict[str, Any] = {
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "config": asdict(cfg),
    }

    # ----- Evaluate GNN AF checkpoints (fixes your hidden_dim mismatch) -----
    feature_mode: FeatureMode = cast(FeatureMode, "AF")

    # Load hp that produced the checkpoints
    hp_pack = _load_hp_from_gnn_metrics(paths, feature_mode)
    gcn_hp = cast(dict[str, Any], hp_pack["gcn_hp"])
    sgcn_hp = cast(dict[str, Any], hp_pack["skip_gcn_hp"])

    # Rebuild graph exactly like phase03 for AF
    data, train_steps, test_steps = _build_graph_for_mode(
        df_full=df_full,
        df_labeled=df_labeled,
        edges_df=edges_df,
        feature_mode=feature_mode,
        cfg=cfg,
    )

    out["split"] = {"train_steps": train_steps.tolist(), "test_steps": test_steps.tolist()}
    out["graph"] = {
        "num_nodes": int(_require_tensor(data.x, "data.x").size(0)),
        "num_edges": int(_require_tensor(data.edge_index, "data.edge_index").size(1)),
    }

    in_dim = int(_require_tensor(data.x, "data.x").size(1))

    # Checkpoints
    gcn_ckpt = paths.results_dir / "model_artifacts" / f"gcn_{feature_mode}.pt"
    sgcn_ckpt = paths.results_dir / "model_artifacts" / f"skip_gcn_{feature_mode}.pt"
    if not gcn_ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {gcn_ckpt}")
    if not sgcn_ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {sgcn_ckpt}")

    # Instantiate with correct hidden_dim/dropout, then load
    gcn = _instantiate_gcn(in_dim=in_dim, hp=gcn_hp)
    gcn_state = torch.load(gcn_ckpt, map_location="cpu")
    gcn.load_state_dict(cast(dict[str, Tensor], gcn_state))
    out[f"gcn_{feature_mode}"] = {
        "hp": gcn_hp,
        "metrics": _eval_loaded_model(gcn, data, device),
    }

    sgcn = _instantiate_skipgcn(in_dim=in_dim, hp=sgcn_hp)
    sgcn_state = torch.load(sgcn_ckpt, map_location="cpu")
    sgcn.load_state_dict(cast(dict[str, Tensor], sgcn_state))
    out[f"skip_gcn_{feature_mode}"] = {
        "hp": sgcn_hp,
        "metrics": _eval_loaded_model(sgcn, data, device),
    }

    # Save report
    report_path = paths.results_dir / "metrics" / "phase05_eval_report.json"
    report_path.write_text(
    json.dumps(json_safe(out), indent=2),
    encoding="utf-8",
    )



if __name__ == "__main__":
    main()
