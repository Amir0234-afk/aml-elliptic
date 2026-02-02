# src/phase02_eda.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.config import ExperimentConfig, Paths
from src.data import load_edges
from src.repro import set_seed
from src.runlog import write_run_manifest
from src.viz import (
    save_class_distribution,
    save_degree_distribution,
    save_illicit_ratio_over_time,
    save_label_counts_over_time,
    save_nodes_per_timestep,
    save_top_feature_mean_diff,
)


def _load_csv(paths: Paths, name: str) -> pd.DataFrame:
    p = paths.processed_dir / name
    if not p.exists():
        raise FileNotFoundError(f"Missing processed file: {p}")
    return pd.read_csv(p)


def main() -> None:
    paths = Paths()
    cfg = ExperimentConfig()
    set_seed(cfg.seed, deterministic_torch=False)

    # inputs
    df_full = _load_csv(paths, "elliptic_full.csv")       # contains class in {-1,0,1}
    df_labeled = _load_csv(paths, "elliptic_labeled.csv") # contains class in {0,1}
    edges_df = load_edges(paths.raw_dir)

    # logging
    write_run_manifest(
        paths.results_dir / "logs",
        phase="02",
        cfg=cfg,
        data_files=[
            paths.raw_dir / "elliptic_txs_features.csv",
            paths.raw_dir / "elliptic_txs_edgelist.csv",
            paths.raw_dir / "elliptic_txs_classes.csv",
            paths.processed_dir / "elliptic_full.csv",
            paths.processed_dir / "elliptic_labeled.csv",
        ],
        extra={"note": "phase 02: EDA + visualization export"},
    )

    out_dir = paths.results_dir / "visualizations"
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

    # ----- Metrics summary (Phase 02 report will cite these numbers) -----
    summary: Dict[str, Any] = {
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "full_rows": int(len(df_full)),
        "labeled_rows": int(len(df_labeled)),
        "unlabeled_rows": int((df_full["class"] == -1).sum()),
        "time_steps_full": int(df_full["time_step"].nunique()),
        "time_steps_labeled": int(df_labeled["time_step"].nunique()),
        "label_counts_labeled": {str(k): int(v) for k, v in df_labeled["class"].value_counts().to_dict().items()},
        "edges_rows": int(len(edges_df)),
        "unique_nodes_in_edges": int(pd.unique(pd.concat([edges_df["txId1"], edges_df["txId2"]])).shape[0]),
    }

    (paths.results_dir / "metrics" / "phase02_eda_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
