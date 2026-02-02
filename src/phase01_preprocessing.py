# src/phase01_preprocessing.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import Paths
from src.data import build_full_dataset, load_elliptic
from src.runlog import write_run_manifest


def _ensure_dirs(paths: Paths) -> None:
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "logs").mkdir(parents=True, exist_ok=True)


def _save_df(df: pd.DataFrame, parquet_path: Path, csv_path: Path) -> str:
    try:
        df.to_parquet(parquet_path, index=False)
        return str(parquet_path)
    except Exception:
        df.to_csv(csv_path, index=False)
        return str(csv_path)


def main() -> None:
    paths = Paths()
    _ensure_dirs(paths)

    write_run_manifest(
        paths.results_dir / "logs",
        phase="01",
        cfg={"note": "preprocessing: export elliptic_full + elliptic_labeled + elliptic_edges"},
        data_files=[
            paths.raw_dir / "elliptic_txs_features.csv",
            paths.raw_dir / "elliptic_txs_edgelist.csv",
            paths.raw_dir / "elliptic_txs_classes.csv",
        ],
    )

    loaded = load_elliptic(paths.raw_dir)

    df_full = build_full_dataset(loaded.features, loaded.classes)

    # Sanity: one row per txId
    if df_full["txId"].nunique() != df_full.shape[0]:
        raise RuntimeError("Expected 1 row per txId in full dataset.")

    df_labeled = df_full[df_full["class"] != -1].copy()
    if df_labeled.empty:
        raise RuntimeError("No labeled data found after filtering. Check class parsing.")

    # Save full dataset
    full_saved_as = _save_df(
        df_full,
        paths.processed_dir / "elliptic_full.parquet",
        paths.processed_dir / "elliptic_full.csv",
    )

    # Save labeled-only dataset
    labeled_saved_as = _save_df(
        df_labeled,
        paths.processed_dir / "elliptic_labeled.parquet",
        paths.processed_dir / "elliptic_labeled.csv",
    )

    # Save edges (as-loaded)
    edges_saved_as = _save_df(
        loaded.edges,
        paths.processed_dir / "elliptic_edges.parquet",
        paths.processed_dir / "elliptic_edges.csv",
    )

    feat_cols = [c for c in df_full.columns if c.startswith("feat_")]

    summary = {
        "saved_full_dataset": full_saved_as,
        "saved_labeled_dataset": labeled_saved_as,
        "saved_edges": edges_saved_as,
        "raw_features_rows": int(loaded.features.shape[0]),
        "raw_edges_rows": int(loaded.edges.shape[0]),
        "raw_classes_rows": int(loaded.classes.shape[0]),
        "full_rows": int(df_full.shape[0]),
        "labeled_rows": int(df_labeled.shape[0]),
        "unknown_rows": int((df_full["class"] == -1).sum()),
        "label_counts_full": {str(k): int(v) for k, v in df_full["class"].value_counts().to_dict().items()},
        "label_counts_labeled": {str(k): int(v) for k, v in df_labeled["class"].value_counts().to_dict().items()},
        "time_steps_min": int(df_full["time_step"].min()),
        "time_steps_max": int(df_full["time_step"].max()),
        "time_steps_unique": int(df_full["time_step"].nunique()),
        "num_feature_cols": int(len(feat_cols)),
    }

    (paths.results_dir / "metrics" / "phase01_data_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
