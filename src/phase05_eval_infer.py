# src/phase05_eval_infer.py
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, cast

import joblib
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data

from src.config import ExperimentConfig, Paths
from src.data import FeatureMode, get_feature_cols, load_edges
from src.eval import compute_metrics, threshold_predictions
from src.gnn import GCN, SkipGCN
from src.graph_data import build_graph
from src.repro import set_seed
from src.runlog import write_run_manifest
from src.viz import save_final_f1_summary_bar


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

def _ensure_dirs(paths: Paths) -> None:
    (paths.results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "logs").mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "visualizations" / "phase05").mkdir(parents=True, exist_ok=True)


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
) -> tuple[Data, np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds the exact same kind of graph bundle as phase03 (normalization, undirected, val split within train).
    Returns (data, train_only_steps, val_steps, test_steps).
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
    return data, bundle.train_steps, bundle.val_steps, bundle.test_steps


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

def _find_phase04_backend_name(paths: Paths) -> str | None:
    # Prefer the existing phase04 index file name: phase04_tuning_index_{backend}.json
    mdir = paths.results_dir / "metrics"
    cands = sorted(mdir.glob("phase04_tuning_index_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        return None
    name = cands[0].stem.replace("phase04_tuning_index_", "")
    return name or None

def _predict_proba_pos(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        p = np.asarray(p)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1].astype(float, copy=False)
        return p.astype(float, copy=False)
    raise TypeError(f"Model of type {type(model).__name__} has no predict_proba().")

def _maybe_write_df(df: pd.DataFrame, out_base: Path) -> str:
    # Prefer parquet; fallback to csv if parquet engine missing.
    try:
        p = out_base.with_suffix(".parquet")
        df.to_parquet(p, index=False)
        return str(p)
    except Exception:
        c = out_base.with_suffix(".csv")
        df.to_csv(c, index=False)
        return str(c)

def _eval_phase04_tabular(
    *,
    df_full: pd.DataFrame,
    df_labeled: pd.DataFrame,
    feature_mode: FeatureMode,
    paths: Paths,
    cfg: ExperimentConfig,
    backend_name: str,
    save_predictions: bool,
) -> dict[str, Any]:
    # Recreate the same temporal test split used in phase04
    train_steps, test_steps = _temporal_train_test_steps(df_labeled, cfg.train_ratio)
    feat_cols = get_feature_cols(df_full, feature_mode)

    df_test = df_labeled[df_labeled["time_step"].isin(test_steps)].copy()
    X_test = np.nan_to_num(df_test[feat_cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_test = df_test["class"].to_numpy(dtype=np.int64)

    art = paths.results_dir / "model_artifacts" / "phase04"

    out: dict[str, Any] = {
        "feature_mode": str(feature_mode),
        "backend": str(backend_name),
        "split": {"train_steps": train_steps.tolist(), "test_steps": test_steps.tolist()},
    }

    # ---- LR (with saved (mu,sd) scaler + saved threshold) ----
    lr_path = art / f"lr_{feature_mode}_{backend_name}.joblib"
    sc_path = art / f"lr_scaler_{feature_mode}.joblib"
    th_path = art / f"lr_threshold_{feature_mode}_{backend_name}.json"
    if lr_path.exists() and sc_path.exists() and th_path.exists():
        lr = joblib.load(lr_path)
        ss = cast(dict[str, Any], joblib.load(sc_path))  # {"mu":..., "sd":...}
        thr = float(_read_json(th_path)["threshold"])
        mu = np.asarray(ss["mu"], dtype=np.float32)
        sd = np.asarray(ss["sd"], dtype=np.float32)
        Xte = (X_test - mu) / sd
        p = _predict_proba_pos(lr, Xte)
        yhat = threshold_predictions(p, thr)
        m = compute_metrics(y_test, yhat)
        out["lr"] = {"threshold": thr, "metrics": m.__dict__}

        if save_predictions:
            dfp = df_test[["txId", "time_step", "class"]].copy()
            dfp["p_illicit"] = p.astype(float)
            dfp["y_pred"] = yhat.astype(int)
            saved = _maybe_write_df(dfp, paths.results_dir / "predictions" / f"phase05_tabular_lr_{feature_mode}_{backend_name}")
            out["lr"]["predictions_path"] = saved

    # ---- RF (raw features + saved threshold) ----
    rf_path = art / f"rf_{feature_mode}_{backend_name}.joblib"
    th2_path = art / f"rf_threshold_{feature_mode}_{backend_name}.json"
    if rf_path.exists() and th2_path.exists():
        rf = joblib.load(rf_path)
        thr2 = float(_read_json(th2_path)["threshold"])
        p2 = _predict_proba_pos(rf, X_test)
        yhat2 = threshold_predictions(p2, thr2)
        m2 = compute_metrics(y_test, yhat2)
        out["rf"] = {"threshold": thr2, "metrics": m2.__dict__}

        if save_predictions:
            dfp2 = df_test[["txId", "time_step", "class"]].copy()
            dfp2["p_illicit"] = p2.astype(float)
            dfp2["y_pred"] = yhat2.astype(int)
            saved2 = _maybe_write_df(dfp2, paths.results_dir / "predictions" / f"phase05_tabular_rf_{feature_mode}_{backend_name}")
            out["rf"]["predictions_path"] = saved2

    return out

def _gnn_infer_all_nodes(model: torch.nn.Module, data: Data, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    # Returns (p_illicit, y_pred) for ALL nodes in Data (aligned with node order).
    model = model.to(device)
    model.eval()
    d = data.to(device)
    x = _require_tensor(d.x, "data.x")
    ei = _require_tensor(d.edge_index, "data.edge_index")
    with torch.no_grad():
        logits = model(x, ei)
        prob = torch.softmax(logits, dim=1)[:, 1]
        yhat = logits.argmax(dim=1)
    return prob.detach().cpu().numpy().astype(float), yhat.detach().cpu().numpy().astype(int)



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

    _ensure_dirs(paths)

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
        extra={"note": "phase 05: final eval (+ optional inference exports). GNN: load hp from gnn_*.json to avoid shape mismatch; Tabular: evaluate phase04 tuned artifacts"},
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

    # ----- Evaluate tuned TABULAR (phase04) -----
    p04_backend = _find_phase04_backend_name(paths)
    if p04_backend is not None:
        out["phase04_tabular_eval"] = {}
        for mode in (cast(FeatureMode, "AF"), cast(FeatureMode, "LF")):
            out["phase04_tabular_eval"][str(mode)] = _eval_phase04_tabular(
                df_full=df_full,
                df_labeled=df_labeled,
                feature_mode=mode,
                paths=paths,
                cfg=cfg,
                backend_name=p04_backend,
                save_predictions=True,
            )

    # ----- Evaluate GNN checkpoints (load hp from phase03 gnn_*.json) -----
    feature_mode: FeatureMode = cast(FeatureMode, "AF")  # matches cfg.gnn_feature_modes default


    # Load hp that produced the checkpoints
    hp_pack = _load_hp_from_gnn_metrics(paths, feature_mode)
    gcn_hp = cast(dict[str, Any], hp_pack["gcn_hp"])
    sgcn_hp = cast(dict[str, Any], hp_pack["skip_gcn_hp"])

    # Rebuild graph exactly like phase03 for AF
    data, train_only_steps, val_steps, test_steps = _build_graph_for_mode(
        df_full=df_full,
        df_labeled=df_labeled,
        edges_df=edges_df,
        feature_mode=feature_mode,
        cfg=cfg,
    )

    out["split"] = {
        "train_steps": train_only_steps.tolist(),
        "val_steps": val_steps.tolist(),
        "test_steps": test_steps.tolist(),
    }
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

    # Optional: export GNN predictions for ALL nodes (includes unlabeled nodes with class=-1)
    # Alignment: build_graph sorts by txId, so sort df_full by txId to match node order.
    df_full_sorted = df_full.sort_values("txId").reset_index(drop=True)
    for name, model in [(f"gcn_{feature_mode}", gcn), (f"skip_gcn_{feature_mode}", sgcn)]:
        p_all, yhat_all = _gnn_infer_all_nodes(model, data, device)
        dfp = df_full_sorted[["txId", "time_step", "class"]].copy()
        dfp["p_illicit"] = p_all.astype(float)
        dfp["y_pred"] = yhat_all.astype(int)
        saved = _maybe_write_df(dfp, paths.results_dir / "predictions" / f"phase05_gnn_{name}")
        out[name]["predictions_path"] = saved

    # ---- Phase05 final summary plot ----
    f1_map: dict[str, float] = {}

    # Tabular (phase04 tuned artifacts)
    if "phase04_tabular_eval" in out:
        te = out["phase04_tabular_eval"]
        for mode in ("AF", "LF"):
            if mode in te:
                if "lr" in te[mode]:
                    f1_map[f"LR_{mode}_{te[mode]['backend']}"] = float(te[mode]["lr"]["metrics"]["f1_illicit"])
                if "rf" in te[mode]:
                    f1_map[f"RF_{mode}_{te[mode]['backend']}"] = float(te[mode]["rf"]["metrics"]["f1_illicit"])

    # GNN
    if f"gcn_{feature_mode}" in out:
        f1_map[f"GCN_{feature_mode}"] = float(out[f"gcn_{feature_mode}"]["metrics"]["f1_illicit"])
    if f"skip_gcn_{feature_mode}" in out:
        f1_map[f"SkipGCN_{feature_mode}"] = float(out[f"skip_gcn_{feature_mode}"]["metrics"]["f1_illicit"])

    viz_dir = paths.results_dir / "visualizations" / "phase05"
    plot_path = save_final_f1_summary_bar(f1_map, viz_dir)
    out["final_summary_plot"] = str(plot_path)


    # Save report
    report_path = paths.results_dir / "metrics" / "phase05_eval_report.json"
    report_path.write_text(
        json.dumps(json_safe(out), indent=2),
        encoding="utf-8",
    )



if __name__ == "__main__":
    main()
