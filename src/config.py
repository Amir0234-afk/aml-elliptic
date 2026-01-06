from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Paths:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    results_dir: Path = Path("results")
    reports_dir: Path = Path("reports")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42

    # Temporal split
    train_ratio: float = 0.70
    val_ratio_within_train: float = 0.10  # last 10% of train steps reserved as val

    # Baselines
    rf_n_estimators: int = 200
    rf_max_features: Literal["sqrt", "log2"] | float = "sqrt"

    # GNN training
    enable_gnn: bool = True
    gnn_feature_modes: tuple[str, ...] = ("AF",)  # run GNN on AF by default
    gcn_hidden_dim: int = 100
    gcn_dropout: float = 0.5
    lr: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 20
    device: str = "cpu"
