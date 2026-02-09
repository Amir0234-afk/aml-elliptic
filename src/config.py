# src/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


def _truthy_env(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def resolve_device(requested: str | None) -> str:
    """
    Resolve device string deterministically.

    Priority:
      1) AML_DEVICE env var (if set)
      2) requested argument
      3) "auto"

    Accepted:
      - "auto"
      - "cpu"
      - "cuda" / "cuda:N"
      - "mps" (macOS)

    If "auto": picks "cuda" if torch.cuda.is_available() else "cpu".
    """
    dev = (requested or "auto").strip().lower()

    if dev == "auto":
        try:
            import torch  # local import to avoid hard dependency at import time
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    if dev == "cpu":
        return "cpu"

    if dev == "mps":
        return "mps"

    if dev == "cuda":
        return "cuda"

    if dev.startswith("cuda:"):
        suffix = dev.split("cuda:", 1)[1]
        if suffix.isdigit():
            return dev

    # fallback safe
    return "cpu"


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
    val_ratio_within_train: float = 0.10  # last 10% of train steps reserved as val (GNN)

    # Baselines (used as defaults; HPO can override)
    rf_n_estimators: int = 200
    rf_max_features: Literal["sqrt", "log2"] | float = "sqrt"

    # Tabular tuning
    tabular_tune: bool = True
    tabular_tune_trials: int = 30
    tabular_cv_splits: int = 5
    tabular_cv_val_steps: int = 1

    # Logistic Regression runtime controls
    lr_max_iter: int = 1500
    lr_tol: float = 1e-3

    # GNN training
    enable_gnn: bool = True
    gnn_feature_modes: tuple[str, ...] = ("AF",)

    # GNN tuning
    gnn_tune_trials: int = 10
    gnn_tune_class_weight: bool = True

    # Base GNN training budget
    epochs: int = 200
    patience: int = 20

    # Default values used in search grids
    gcn_hidden_dim: int = 100
    gcn_dropout: float = 0.5
    lr: float = 1e-3
    weight_decay: float = 5e-4

    # Device selection:
    #   - default "auto" => cuda if available else cpu
    #   - can override via env: AML_DEVICE=cpu|cuda|cuda:0|auto
    device: str = "auto"

    # Embedding augmentation: RF on (features ⊕ GNN hidden embedding)
    enable_embedding_aug: bool = True
    embedding_aug_model: Literal["gcn", "skip_gcn"] = "gcn"
    embedding_aug_rf_n_estimators: int = 500

    def __post_init__(self) -> None:
        # Resolve device from env/request
        env_dev = os.getenv("AML_DEVICE")
        resolved = resolve_device(env_dev if env_dev is not None else self.device)
        object.__setattr__(self, "device", resolved)

        # Optional fast mode (no code changes elsewhere):
        #   AML_FAST=1 makes runs much shorter.
        if _truthy_env("AML_FAST", "0"):
            object.__setattr__(self, "tabular_tune_trials", min(self.tabular_tune_trials, 10))
            object.__setattr__(self, "gnn_tune_trials", min(self.gnn_tune_trials, 4))
            object.__setattr__(self, "epochs", min(self.epochs, 120))
            object.__setattr__(self, "patience", min(self.patience, 10))
            object.__setattr__(self, "enable_embedding_aug", False)
