# src/phase01_preprocessing.py
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import Paths
from src.data import build_full_dataset, load_elliptic
from src.runlog import write_run_manifest


SCHEMA_VERSION = "phase01_v3"

def _ensure_dirs(paths: Paths) -> None:
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "logs").mkdir(parents=True, exist_ok=True)


def _sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _file_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "bytes": int(st.st_size),
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": _sha256_file(path),
    }

def _coerce_dtypes_full(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce stable dtypes for downstream reproducibility and speed.
    - txId, time_step: int64
    - class: int64 (values in {-1,0,1})
    - feat_*: float32
    """
    df = df_full.copy()
    df["txId"] = df["txId"].astype(np.int64)
    df["time_step"] = df["time_step"].astype(np.int64)
    df["class"] = df["class"].astype(np.int64)

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    if feat_cols:
        df.loc[:, feat_cols] = df.loc[:, feat_cols].astype(np.float32)
    return df


def _truthy_env(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _save_df(df: pd.DataFrame, parquet_path: Path, csv_path: Path) -> tuple[str, dict[str, str]]:
    """
    Save parquet if possible, otherwise csv.
    Returns (saved_path, info_dict).
    info_dict contains parquet failure info if applicable, e.g. {"parquet_error": "..."}.
    """
    info: dict[str, str] = {}
    try:
        df.to_parquet(parquet_path, index=False)
        info["format"] = "parquet"
        return str(parquet_path), info
    except Exception as e:
        info["format"] = "csv"
        info["parquet_error"] = f"{type(e).__name__}: {e}"
        df.to_csv(csv_path, index=False)
        return str(csv_path), info


def _assert_data_contract(df_full: pd.DataFrame, feat_cols: list[str]) -> None:
    required_cols = {"txId", "time_step", "class"}
    missing = required_cols - set(df_full.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in full dataset: {sorted(missing)}")

    if df_full["txId"].nunique() != df_full.shape[0]:
        raise RuntimeError("Expected 1 row per txId in full dataset.")

    ts = df_full["time_step"]
    if ts.isna().any():
        raise RuntimeError("Found NaN values in time_step column.")
    ts_min = int(ts.min())
    ts_max = int(ts.max())
    ts_n = int(ts.nunique())
    if ts_min != 1 or ts_max != 49 or ts_n != 49:
        raise RuntimeError(
            f"Unexpected time_step values: min={ts_min}, max={ts_max}, unique={ts_n} (expected 1-49)."
        )

    if not feat_cols:
        raise RuntimeError("No feature columns found (expected columns starting with 'feat_').")

    X = df_full[feat_cols].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite values found in feature matrix after preprocessing (NaN/inf).")

def _assert_node_index(tx_ids: np.ndarray) -> None:
    if tx_ids.ndim != 1:
        raise RuntimeError(f"Expected 1D array of txIds for node index. Got shape {tx_ids.shape}.")
    if tx_ids.size ==0:
        raise RuntimeError("Empty txId array for node index.")
    # preprocessing sorts by txId; ensure stable monotonic order
    if not np.all(tx_ids[:-1] < tx_ids[1:]):
        raise RuntimeError("tx_ids is not sorted ascending; node ordering is not deterministic.")
    if np.unique(tx_ids).size != tx_ids.size:
        raise RuntimeError("Duplicate txIds found in node index; expected unique txIds.")

def _edge_stats(edges: pd.DataFrame, node_ids: pd.Series) -> dict[str, int | float]:
    """
    Basic graph diagnostics on the RAW directed edge list.
    Degree stats are computed on the induced subgraph restricted to node_ids,
    treating the edge list as undirected for degree magnitude diagnostics.
    """
    src = edges["txId1"].to_numpy(dtype=np.int64, copy=False)
    dst = edges["txId2"].to_numpy(dtype=np.int64, copy=False)
    nodes = node_ids.to_numpy(dtype=np.int64, copy=False)

    keep = np.isin(src, nodes) & np.isin(dst, nodes)
    src = src[keep]
    dst = dst[keep]

    m = int(src.shape[0])
    self_loops = int((src == dst).sum())

    if m:
        pairs = np.vstack([src, dst]).T
        uniq_pairs = np.unique(pairs, axis=0)
        dup = int(m - uniq_pairs.shape[0])
    else:
        dup = 0

    if m:
        deg_counts = pd.Series(np.concatenate([src, dst])).value_counts()
        deg_min = int(deg_counts.min())
        deg_max = int(deg_counts.max())
        deg_mean = float(deg_counts.mean())
        deg_median = float(deg_counts.median())
        isolated = int(len(nodes) - int(deg_counts.shape[0]))
    else:
        deg_min = 0
        deg_max = 0
        deg_mean = 0.0
        deg_median = 0.0
        isolated = int(len(nodes))

    return {
        "edges_kept": m,
        "self_loops": self_loops,
        "duplicate_edges_directed": dup,
        "deg_min": deg_min,
        "deg_max": deg_max,
        "deg_mean": deg_mean,
        "deg_median": deg_median,
        "isolated_nodes": isolated,
    }

def _edge_stats_directed(edges: pd.DataFrame, node_ids: pd.Series) -> dict[str, int | float]:
    """
    Directed diagnostics on edge list restricted to node_ids:
    - in-degree and out-degree stats
    - number of nodes with zero in/out degree
    """
    nodes = node_ids.to_numpy(dtype=np.int64, copy=False)
    src = edges["txId1"].to_numpy(dtype=np.int64, copy=False)
    dst = edges["txId2"].to_numpy(dtype=np.int64, copy=False)

    keep = np.isin(src, nodes) & np.isin(dst, nodes)
    src = src[keep]
    dst = dst[keep]

    m = int(src.shape[0])
    if m == 0:
        return {
            "edges_kept": 0,
            "out_deg_min": 0,
            "out_deg_max": 0,
            "out_deg_mean": 0.0,
            "out_deg_median": 0.0,
            "in_deg_min": 0,
            "in_deg_max": 0,
            "in_deg_mean": 0.0,
            "in_deg_median": 0.0,
            "zero_out_deg_nodes": int(nodes.shape[0]),
            "zero_in_deg_nodes": int(nodes.shape[0]),
        }
    out_counts = pd.Series(src).value_counts()
    in_counts = pd.Series(dst).value_counts()

    zero_out = int(nodes.shape[0] - int(out_counts.shape[0]))
    zero_in = int(nodes.shape[0] - int(in_counts.shape[0]))
    return {
        "edges_kept": m,
        "out_deg_min": int(out_counts.min()),
        "out_deg_max": int(out_counts.max()),
        "out_deg_mean": float(out_counts.mean()),
        "out_deg_median": float(out_counts.median()),
        "in_deg_min": int(in_counts.min()),
        "in_deg_max": int(in_counts.max()),
        "in_deg_mean": float(in_counts.mean()),
        "in_deg_median": float(in_counts.median()),
        "zero_out_deg_nodes": zero_out,
        "zero_in_deg_nodes": zero_in,
    }



def _filter_edges_to_nodes(edges: pd.DataFrame, node_ids: pd.Series) -> pd.DataFrame:
    nodes = node_ids.to_numpy(dtype=np.int64, copy=False)
    src = edges["txId1"].to_numpy(dtype=np.int64, copy=False)
    dst = edges["txId2"].to_numpy(dtype=np.int64, copy=False)
    keep = np.isin(src, nodes) & np.isin(dst, nodes)
    out = edges.loc[keep, ["txId1", "txId2"]].copy()
    # Determinize: drop duplicates and sort for stable byte-identical exports
    out["txId1"] = out["txId1"].astype(np.int64)
    out["txId2"] = out["txId2"].astype(np.int64)
    out = out.drop_duplicates()
    out = out.sort_values(["txId1", "txId2"]).reset_index(drop=True)
    return out

def _save_node_index(paths: Paths, tx_ids: np.ndarray) -> tuple[str, dict[str, str]]:
    """
    Save canonical node ordering: txId -> node_idx (0..N-1).
    This prevents accidental re-ordering differences across scripts/runs.
    """
    df = pd.DataFrame({"txId": tx_ids.astype(np.int64), "node_idx": np.arange(tx_ids.shape[0], dtype=np.int64)})
    return _save_df(
        df,
        paths.processed_dir / "node_index.parquet",
        paths.processed_dir / "node_index.csv",
    )


def _edge_coverage(edges: pd.DataFrame, node_ids: pd.Series) -> dict[str, float | int]:
    nodes = node_ids.to_numpy(dtype=np.int64, copy=False)
    src = edges["txId1"].to_numpy(dtype=np.int64, copy=False)
    dst = edges["txId2"].to_numpy(dtype=np.int64, copy=False)

    keep = np.isin(src, nodes) & np.isin(dst, nodes)
    if not keep.any():
        return {
            "unique_nodes_in_edges_induced": 0,
            "missing_nodes_from_edges": int(nodes.shape[0]),
            "coverage_ratio": 0.0,
        }

    src_k = src[keep]
    dst_k = dst[keep]
    unique_nodes = int(np.unique(np.concatenate([src_k, dst_k])).shape[0])
    missing = int(nodes.shape[0] - unique_nodes)
    return {
        "unique_nodes_in_edges_induced": unique_nodes,
        "missing_nodes_from_edges": missing,
        "coverage_ratio": float(unique_nodes / nodes.shape[0]),
    }


def main() -> None:
    paths = Paths()
    _ensure_dirs(paths)

    strict = _truthy_env("AML_STRICT_ELLIPTIC", "1")

    write_run_manifest(
        paths.results_dir / "logs",
        phase="01",
        cfg={"note": "preprocessing: export elliptic_full + elliptic_labeled + elliptic_edges(_kept)"},
        data_files=[
            paths.raw_dir / "elliptic_txs_features.csv",
            paths.raw_dir / "elliptic_txs_edgelist.csv",
            paths.raw_dir / "elliptic_txs_classes.csv",
        ],
        extra={"schema_version": SCHEMA_VERSION},
    )

    loaded = load_elliptic(paths.raw_dir)

    # Strict validation: features file should be 167 cols after header=None read
    if strict and loaded.features.shape[1] != 167:
        raise RuntimeError(
            f"Unexpected raw features column count: {loaded.features.shape[1]} (expected 167). "
            "Set AML_STRICT_ELLIPTIC=0 to bypass."
        )

    df_full = build_full_dataset(loaded.features, loaded.classes)
    df_full = _coerce_dtypes_full(df_full)

    # canonical node ordering (txId sorted by preprocessing)
    tx_ids = df_full["txId"].to_numpy(dtype=np.int64, copy=False)
    _assert_node_index(tx_ids)

    # Save canonical node index (helps downstream graph building stay deterministic)
    node_index_saved_as = _save_node_index(paths, tx_ids)

    feat_cols = [c for c in df_full.columns if c.startswith("feat_")]
    _assert_data_contract(df_full, feat_cols)

    df_labeled = df_full[df_full["class"] != -1].copy()
    if df_labeled.empty:
        raise RuntimeError("No labeled data found after filtering. Check class parsing.")

    edges_kept = _filter_edges_to_nodes(loaded.edges, df_full["txId"])

    full_saved_as = _save_df(
        df_full,
        paths.processed_dir / "elliptic_full.parquet",
        paths.processed_dir / "elliptic_full.csv",
    )
    labeled_saved_as = _save_df(
        df_labeled,
        paths.processed_dir / "elliptic_labeled.parquet",
        paths.processed_dir / "elliptic_labeled.csv",
    )
    edges_saved_as = _save_df(
        loaded.edges,
        paths.processed_dir / "elliptic_edges.parquet",
        paths.processed_dir / "elliptic_edges.csv",
    )
    edges_kept_saved_as = _save_df(
        edges_kept,
        paths.processed_dir / "elliptic_edges_kept.parquet",
        paths.processed_dir / "elliptic_edges_kept.csv",
    )

    raw_class_series = df_full["class_raw"].astype(str)
    raw_lower = raw_class_series.str.strip().str.lower()
    raw_counts = raw_lower.value_counts(dropna=False).to_dict()

    classes_missing_for_txid = int(df_full["class_raw"].isna().sum())
    unknown_string_count = int((raw_lower == "unknown").sum())

    edge_stats_raw_induced = _edge_stats(loaded.edges, df_full["txId"])
    edge_stats_kept_induced = _edge_stats(edges_kept, df_full["txId"])
    edge_stats_raw_directed = _edge_stats_directed(loaded.edges, df_full["txId"])
    edge_stats_kept_directed = _edge_stats_directed(edges_kept, df_full["txId"])
    edge_coverage = _edge_coverage(loaded.edges, df_full["txId"])

    raw_inputs = {
        "features": _file_meta(paths.raw_dir / "elliptic_txs_features.csv"),
        "edgelist": _file_meta(paths.raw_dir / "elliptic_txs_edgelist.csv"),
        "classes": _file_meta(paths.raw_dir / "elliptic_txs_classes.csv"),
    }

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "saved_full_dataset": full_saved_as,
        "saved_labeled_dataset": labeled_saved_as,
        "saved_edges_raw": edges_saved_as,
        "saved_edges_kept": edges_kept_saved_as,
        "saved_node_index": node_index_saved_as,
        "raw_inputs": raw_inputs,
        "raw_features_rows": int(loaded.features.shape[0]),
        "raw_features_cols": int(loaded.features.shape[1]),
        "raw_edges_rows": int(loaded.edges.shape[0]),
        "raw_classes_rows": int(loaded.classes.shape[0]),
        "full_rows": int(df_full.shape[0]),
        "labeled_rows": int(df_labeled.shape[0]),
        "unknown_rows": int((df_full["class"] == -1).sum()),
        "label_counts_full_mapped": {str(k): int(v) for k, v in df_full["class"].value_counts().to_dict().items()},
        "label_counts_labeled_mapped": {str(k): int(v) for k, v in df_labeled["class"].value_counts().to_dict().items()},
        "label_counts_full_raw": {str(k): int(v) for k, v in raw_counts.items()},
        "classes_missing_for_txid": classes_missing_for_txid,
        "unknown_string_count": unknown_string_count,
        "time_steps_min": int(df_full["time_step"].min()),
        "time_steps_max": int(df_full["time_step"].max()),
        "time_steps_unique": int(df_full["time_step"].nunique()),
        "num_feature_cols": int(len(feat_cols)),
        "strict_elliptic_enabled": bool(strict),
        "edges": {
            "raw_rows": int(loaded.edges.shape[0]),
            "kept_rows": int(edges_kept.shape[0]),
            "dropped_rows": int(loaded.edges.shape[0] - edges_kept.shape[0]),
        },
        "edge_stats_raw_induced": edge_stats_raw_induced,
        "edge_stats_kept_induced": edge_stats_kept_induced,
        "edge_stats_raw_directed": edge_stats_raw_directed,
        "edge_stats_kept_directed": edge_stats_kept_directed,
        "edge_coverage": edge_coverage,
    }

    (paths.results_dir / "metrics" / "phase01_data_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
