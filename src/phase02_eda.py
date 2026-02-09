# src/phase02_eda.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

from networkx import nodes
import numpy as np
import pandas as pd
from sympy import deg

import src
from src.config import ExperimentConfig, Paths
from src.data import load_edges
from src.repro import set_seed
from src.runlog import write_run_manifest
from src.viz import (
    save_count_over_time,
    save_class_distribution,
    save_degree_distribution,
    save_degree_distribution_from_degrees,
    save_illicit_ratio_over_time,
    save_label_counts_over_time,
    save_nodes_per_timestep,
    save_top_feature_mean_diff,
    save_metric_over_time
)


def _load_table(paths: Paths, stem: str) -> Tuple[pd.DataFrame, Path, str]:
    """
    Parquet-first loader to avoid stale CSV artifacts (Phase01 v3 writes parquet when possible).
    Returns (df, path_used, fmt).
    """
    p_parq = paths.processed_dir / f"{stem}.parquet"
    p_csv = paths.processed_dir / f"{stem}.csv"
    if p_parq.exists():
        return pd.read_parquet(p_parq), p_parq, "parquet"
    if p_csv.exists():
        return pd.read_csv(p_csv), p_csv, "csv"
    raise FileNotFoundError(f"Missing processed file: {p_parq} or {p_csv}")


def _load_edges_for_eda(paths: Paths) -> Tuple[pd.DataFrame, Path | None, str]:
    """
    Prefer processed deterministic edges (kept) if available; fallback to processed raw; fallback to raw_dir.
    Returns (edges_df, path_used_or_None_if_rawdir_loader, kind).
    """
    candidates: list[tuple[Path, str]] = [
        (paths.processed_dir / "elliptic_edges_kept.parquet", "processed_edges_kept_parquet"),
        (paths.processed_dir / "elliptic_edges_kept.csv", "processed_edges_kept_csv"),
        (paths.processed_dir / "elliptic_edges.parquet", "processed_edges_raw_parquet"),
        (paths.processed_dir / "elliptic_edges.csv", "processed_edges_raw_csv"),
    ]
    for p, kind in candidates:
        if p.exists():
            if p.suffix == ".parquet":
                return pd.read_parquet(p), p, kind
            return pd.read_csv(p), p, kind
    # fallback: raw edgelist (should be avoided for consistency, but kept as escape hatch)
    return load_edges(paths.raw_dir), None, "raw_dir_edges"


def _assert_phase02_contract(df_full: pd.DataFrame, df_labeled: pd.DataFrame, edges_df: pd.DataFrame) -> None:
    # Required columns
    for col in ("txId", "time_step", "class"):
        if col not in df_full.columns:
            raise RuntimeError(f"df_full missing required column: {col}")
        if col not in df_labeled.columns:
            raise RuntimeError(f"df_labeled missing required column: {col}")

    feat_cols = [c for c in df_full.columns if c.startswith("feat_")]
    if not feat_cols:
        raise RuntimeError("No feature columns found in df_full (expected feat_*)")

    # txId uniqueness
    if df_full["txId"].nunique() != len(df_full):
        raise RuntimeError("df_full must have 1 row per txId (txId not unique)")
    if df_labeled["txId"].nunique() != len(df_labeled):
        raise RuntimeError("df_labeled must have 1 row per txId (txId not unique)")

    # class domain
    full_classes = set(pd.unique(df_full["class"]).tolist())
    if not full_classes.issubset({-1, 0, 1}):
        raise RuntimeError(f"Unexpected df_full class values: {sorted(full_classes)} (expected subset of {-1,0,1})")
    lab_classes = set(pd.unique(df_labeled["class"]).tolist())
    if not lab_classes.issubset({0, 1}):
        raise RuntimeError(f"Unexpected df_labeled class values: {sorted(lab_classes)} (expected subset of {0,1})")

    # time_step sanity (Elliptic canonical is 1..49)
    ts = df_full["time_step"]
    if ts.isna().any():
        raise RuntimeError("df_full has NaN time_step")
    ts_min, ts_max, ts_n = int(ts.min()), int(ts.max()), int(ts.nunique())
    if (ts_min, ts_max, ts_n) != (1, 49, 49):
        raise RuntimeError(f"Unexpected df_full time_step range: min={ts_min}, max={ts_max}, unique={ts_n} (expected 1..49)")

    # edge endpoints should be within node set (coverage)
    nodes = df_full["txId"].to_numpy(dtype=np.int64, copy=False)
    src = edges_df["txId1"].to_numpy(dtype=np.int64, copy=False)
    dst = edges_df["txId2"].to_numpy(dtype=np.int64, copy=False)
    ok = np.isin(src, nodes) & np.isin(dst, nodes)
    if not bool(ok.all()):
        bad = int((~ok).sum())
        raise RuntimeError(f"Edges contain endpoints not present in df_full txId set (bad_edges={bad})")
 


def main() -> None:
    paths = Paths()
    cfg = ExperimentConfig()
    set_seed(cfg.seed, deterministic_torch=False)

    # inputs
    df_full, full_path, full_fmt = _load_table(paths, "elliptic_full")          # class in {-1,0,1}
    df_labeled, labeled_path, labeled_fmt = _load_table(paths, "elliptic_labeled")  # class in {0,1}
    edges_df, edges_path, edges_kind = _load_edges_for_eda(paths)

    _assert_phase02_contract(df_full, df_labeled, edges_df)

    # logging
    used_files = [
        paths.raw_dir / "elliptic_txs_features.csv",
        paths.raw_dir / "elliptic_txs_edgelist.csv",
        paths.raw_dir / "elliptic_txs_classes.csv",
        full_path,
        labeled_path,
    ]
    if edges_path is not None:
        used_files.append(edges_path)
    write_run_manifest(
        paths.results_dir / "logs",
        phase="02",
        cfg=cfg,
        data_files=used_files,
        extra={
            "note": "phase 02: EDA + visualization export (parquet-first, processed edges preferred)",
            "full_fmt": full_fmt,
            "labeled_fmt": labeled_fmt,
            "edges_kind": edges_kind,
        },
    )

    # Store Phase02 visuals in a dedicated subdir
    out_dir = paths.results_dir / "visualizations" / "phase02_eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "metrics").mkdir(parents=True, exist_ok=True)

    # ----- Visualizations -----
    save_class_distribution(df_labeled, out_dir)
    save_nodes_per_timestep(df_full, out_dir, fname="nodes_per_timestep_full.png")
    save_nodes_per_timestep(df_labeled, out_dir, fname="nodes_per_timestep_labeled.png")
    save_label_counts_over_time(df_labeled, out_dir)
    save_illicit_ratio_over_time(df_labeled, out_dir)
    save_degree_distribution(edges_df, out_dir)
    save_top_feature_mean_diff(df_labeled, out_dir, top_k=20)

    # Additional Phase02 visuals (counts + fraction)
    g = df_labeled.groupby(["time_step", "class"]).size().unstack(fill_value=0).sort_index()
    g = g.reindex(columns=[0, 1], fill_value=0)
    ts = g.index.to_numpy(dtype=np.int64, copy=False)
    licit_counts = g[0].to_numpy(dtype=np.int64, copy=False)
    illicit_counts = g[1].to_numpy(dtype=np.int64, copy=False)
    save_count_over_time(ts, licit_counts, out_dir, fname="licit_count_over_time.png", title="Licit count over time", ylabel="count")
    save_count_over_time(ts, illicit_counts, out_dir, fname="illicit_count_over_time.png", title="Illicit count over time", ylabel="count")

    full_counts = df_full["time_step"].value_counts().sort_index()
    lab_counts = df_labeled["time_step"].value_counts().sort_index()
    full_counts = full_counts.reindex(full_counts.index, fill_value=0)
    lab_counts = lab_counts.reindex(full_counts.index, fill_value=0)
    frac = (lab_counts / full_counts.replace(0, np.nan)).fillna(0.0)
    save_metric_over_time(
        full_counts.index.to_numpy(dtype=np.int64, copy=False),
        frac.to_numpy(dtype=float, copy=False),
        out_dir,
        fname="labeled_fraction_over_time.png",
        title="Labeled fraction over time",
        ylabel="labeled / total",
    )

    # Degree distributions by group (labeled/unlabeled + licit/illicit)
    deg = pd.concat([edges_df["txId1"], edges_df["txId2"]]).value_counts()
    deg_full = deg.reindex(df_full["txId"], fill_value=0).to_numpy(dtype=np.int64, copy=False)
    labeled_mask = (df_full["class"].to_numpy(dtype=np.int64, copy=False) != -1)
    save_degree_distribution_from_degrees(deg_full[labeled_mask], out_dir, fname_base="degree_labeled", title="Degree distribution (labeled nodes)")
    save_degree_distribution_from_degrees(deg_full[~labeled_mask], out_dir, fname_base="degree_unlabeled", title="Degree distribution (unlabeled nodes)")

    # For licit/illicit, use labeled subset only
    df_lab_tx = df_labeled[["txId", "class"]].copy()
    deg_lab = deg.reindex(df_lab_tx["txId"], fill_value=0).to_numpy(dtype=np.int64, copy=False)
    y_lab = df_lab_tx["class"].to_numpy(dtype=np.int64, copy=False)
    save_degree_distribution_from_degrees(deg_lab[y_lab == 0], out_dir, fname_base="degree_licit", title="Degree distribution (licit nodes)")
    save_degree_distribution_from_degrees(deg_lab[y_lab == 1], out_dir, fname_base="degree_illicit", title="Degree distribution (illicit nodes)")


    # ----- Metrics summary (Phase 02 report will cite these numbers) -----
    # degree percentiles for quick reporting
    d = deg.to_numpy(dtype=np.int64, copy=False)
    p50, p90, p99 = (int(np.percentile(d, q)) for q in (50, 90, 99)) if d.size else (0, 0, 0)
    summary: Dict[str, Any] = {
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "full_path": str(full_path),
            "labeled": str(labeled_path),
            "edges": (str(edges_path) if edges_path is not None else "raw_dir:elliptic_txs_edgelist.csv"),
            "edges_kind": edges_kind,
        },
        "full_rows": int(len(df_full)),
        "labeled_rows": int(len(df_labeled)),
        "unlabeled_rows": int((df_full["class"] == -1).sum()),
        "time_steps_full": int(df_full["time_step"].nunique()),
        "time_steps_labeled": int(df_labeled["time_step"].nunique()),
        "label_counts_labeled": {str(k): int(v) for k, v in df_labeled["class"].value_counts().to_dict().items()},
        "edges_rows": int(len(edges_df)),
        "unique_nodes_in_edges": int(pd.unique(pd.concat([edges_df["txId1"], edges_df["txId2"]])).shape[0]),
        "degree_percentiles": {"p50": p50, "p90": p90, "p99": p99, "max": int(d.max()) if d.size else 0},
    }

    (paths.results_dir / "metrics" / "phase02_eda_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
