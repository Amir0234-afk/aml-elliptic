from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Paths
from src.viz import save_class_distribution, save_nodes_per_timestep


def _load_processed(paths: Paths) -> pd.DataFrame:
    parquet_path = paths.processed_dir / "elliptic_labeled.parquet"
    csv_path = paths.processed_dir / "elliptic_labeled.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError("Processed dataset not found. Run: python -m src.phase01_preprocessing")


def main() -> None:
    paths = Paths()
    out_dir = paths.results_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_processed(paths)
    save_class_distribution(df, out_dir)
    save_nodes_per_timestep(df, out_dir)


if __name__ == "__main__":
    main()
