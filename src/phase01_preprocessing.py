from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import Paths
from src.data import load_elliptic, labeled_only


def _ensure_dirs(paths: Paths) -> None:
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "metrics").mkdir(parents=True, exist_ok=True)


def main() -> None:
    paths = Paths()
    _ensure_dirs(paths)

    loaded = load_elliptic(paths.raw_dir)
    df_labeled = labeled_only(loaded.features, loaded.classes)
    if df_labeled.empty:
        raise RuntimeError("No labeled data found after filtering. Check class parsing.")

    # Save processed dataset (prefer parquet, fallback to CSV)
    parquet_path = paths.processed_dir / "elliptic_labeled.parquet"
    csv_path = paths.processed_dir / "elliptic_labeled.csv"

    saved_as = None
    try:
        df_labeled.to_parquet(parquet_path, index=False)
        saved_as = str(parquet_path)
    except Exception:
        df_labeled.to_csv(csv_path, index=False)
        saved_as = str(csv_path)

    summary = {
        "saved_processed_dataset": saved_as,
        "raw_features_rows": int(loaded.features.shape[0]),
        "raw_edges_rows": int(loaded.edges.shape[0]),
        "raw_classes_rows": int(loaded.classes.shape[0]),
        "labeled_rows": int(df_labeled.shape[0]),
        "label_counts": {str(k): int(v) for k, v in df_labeled["class"].value_counts().to_dict().items()},
        "time_steps_min": int(df_labeled["time_step"].min()),
        "time_steps_max": int(df_labeled["time_step"].max()),
        "time_steps_unique": int(df_labeled["time_step"].nunique()),
    }

    (paths.results_dir / "metrics" / "phase01_data_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
