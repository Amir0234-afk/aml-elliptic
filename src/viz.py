from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_class_distribution(df_labeled: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if df_labeled.empty:
        return

    counts = df_labeled["class"].value_counts().sort_index()
    if counts.empty:
        return

    plt.figure()
    counts.plot(kind="bar")
    plt.title("Label distribution (1=illicit, 2=licit)")
    plt.xlabel("class")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png", dpi=200)
    plt.close()


def save_nodes_per_timestep(df_feat: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = df_feat["time_step"].value_counts().sort_index()
    plt.figure()
    counts.plot(kind="line")
    plt.title("Nodes per time step")
    plt.xlabel("time_step")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "nodes_per_timestep.png", dpi=200)
    plt.close()


def save_model_comparison_bar(model_to_f1: dict[str, float], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(model_to_f1.keys())
    vals = [model_to_f1[k] for k in names]

    plt.figure()
    plt.bar(names, vals)
    plt.title("Illicit F1 by model")
    plt.xlabel("model")
    plt.ylabel("F1 (illicit)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison_f1.png", dpi=200)
    plt.close()
