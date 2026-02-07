# src/viz.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_out(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _to_np_1d(x: object, *, dtype: type | None = None) -> np.ndarray:
    """
    Convert pandas Index/Series/ExtensionArray/list -> numpy 1D array,
    avoiding pandas ExtensionArray types that trigger Pylance/matplotlib stub errors.
    """
    if isinstance(x, (pd.Series, pd.Index)):
        arr = x.to_numpy()
    else:
        arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return np.asarray(arr).reshape(-1)


def save_class_distribution(df_labeled: pd.DataFrame, out_dir: Path) -> None:
    """
    Expects df_labeled['class'] in {0,1} where:
      1 = illicit, 0 = licit
    """
    _ensure_out(out_dir)
    if df_labeled.empty:
        return

    counts = df_labeled["class"].value_counts().sort_index()
    if counts.empty:
        return

    # ensure both classes appear (for consistent plots)
    counts = counts.reindex([0, 1], fill_value=0)

    x = _to_np_1d(counts.index, dtype=int).astype(str)
    y = _to_np_1d(counts, dtype=int)

    plt.figure()
    plt.bar(x, y)
    plt.title("Label distribution (0=licit, 1=illicit)")
    plt.xlabel("class")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png", dpi=200)
    plt.close()


def save_nodes_per_timestep(df_any: pd.DataFrame, out_dir: Path, *, fname: str = "nodes_per_timestep.png") -> None:
    """
    df_any can be labeled-only or full. Plots node count per time_step.
    """
    _ensure_out(out_dir)
    if df_any.empty:
        return

    counts = df_any["time_step"].value_counts().sort_index()
    x = _to_np_1d(counts.index, dtype=int)
    y = _to_np_1d(counts, dtype=int)

    plt.figure()
    plt.plot(x, y)
    plt.title("Nodes per time step")
    plt.xlabel("time_step")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=200)
    plt.close()


def save_label_counts_over_time(df_labeled: pd.DataFrame, out_dir: Path) -> None:
    """
    Stacked area plot of labeled nodes over time (licit vs illicit).
    """
    _ensure_out(out_dir)
    if df_labeled.empty:
        return

    # counts per time_step per class
    g = df_labeled.groupby(["time_step", "class"]).size().unstack(fill_value=0)
    g = g.reindex(columns=[0, 1], fill_value=0).sort_index()

    x = _to_np_1d(g.index, dtype=int)
    licit = _to_np_1d(g[0], dtype=int)
    illicit = _to_np_1d(g[1], dtype=int)

    plt.figure()
    plt.stackplot(x, licit, illicit, labels=["licit (0)", "illicit (1)"])
    plt.title("Labeled nodes over time")
    plt.xlabel("time_step")
    plt.ylabel("count")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "label_counts_over_time.png", dpi=200)
    plt.close()


def save_illicit_ratio_over_time(df_labeled: pd.DataFrame, out_dir: Path) -> None:
    """
    Line plot of illicit ratio among labeled nodes per time_step.
    """
    _ensure_out(out_dir)
    if df_labeled.empty:
        return

    g = df_labeled.groupby(["time_step", "class"]).size().unstack(fill_value=0)
    g = g.reindex(columns=[0, 1], fill_value=0).sort_index()

    total = (g[0] + g[1]).replace(0, np.nan)
    ratio = (g[1] / total).fillna(0.0)

    x = _to_np_1d(ratio.index, dtype=int)
    y = _to_np_1d(ratio, dtype=float)

    plt.figure()
    plt.plot(x, y)
    plt.title("Illicit ratio over time (among labeled)")
    plt.xlabel("time_step")
    plt.ylabel("illicit ratio")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_dir / "illicit_ratio_over_time.png", dpi=200)
    plt.close()


def save_degree_distribution(edges_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Histogram of node degrees for the full edge list (undirected degree).
    """
    _ensure_out(out_dir)
    if edges_df.empty:
        return

    deg = pd.concat([edges_df["txId1"], edges_df["txId2"]]).value_counts()
    d = _to_np_1d(deg, dtype=int)

    plt.figure()
    plt.hist(d, bins=100)
    plt.title("Degree distribution (histogram)")
    plt.xlabel("degree")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "degree_distribution.png", dpi=200)
    plt.close()

    # log view (often more informative for heavy tails)
    plt.figure()
    plt.hist(d, bins=100, log=True)
    plt.title("Degree distribution (log-count)")
    plt.xlabel("degree")
    plt.ylabel("count (log)")
    plt.tight_layout()
    plt.savefig(out_dir / "degree_distribution_log.png", dpi=200)
    plt.close()


def save_degree_distribution_from_degrees(
    degrees: np.ndarray,
    out_dir: Path,
    *,
    fname_base: str,
    title: str,
    bins: int = 100,
) -> None:
    """
    Histogram of node degrees given an explicit degree array.
    Saves both linear and log-count variants.
    """
    _ensure_out(out_dir)
    d = _to_np_1d(degrees, dtype=int)
    if d.size == 0:
        return

    plt.figure()
    plt.hist(d, bins=bins)
    plt.title(title)
    plt.xlabel("degree")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / f"{fname_base}.png", dpi=200)
    plt.close()

    plt.figure()
    plt.hist(d, bins=bins, log=True)
    plt.title(title + " (log-count)")
    plt.xlabel("degree")
    plt.ylabel("count (log)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{fname_base}_log.png", dpi=200)
    plt.close()


def save_count_over_time(
    time_steps: np.ndarray,
    counts: np.ndarray,
    out_dir: Path,
    *,
    fname: str = "count_over_time.png",
    title: str = "Count over time",
    ylabel: str = "count",
) -> None:
    """
    Plot a non-normalized count series over time_step (no [0,1] ylim clamp).
    """
    _ensure_out(out_dir)
    x = _to_np_1d(time_steps, dtype=int)
    y = _to_np_1d(counts, dtype=float)
    if x.size == 0 or y.size == 0:
        return
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]

    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("time_step")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=200)
    plt.close()

 

def save_top_feature_mean_diff(df_labeled: pd.DataFrame, out_dir: Path, *, top_k: int = 20) -> None:
    """
    Bar plot: top-K features by absolute difference in mean between illicit (1) and licit (0).
    """
    _ensure_out(out_dir)
    if df_labeled.empty:
        return

    feat_cols = [c for c in df_labeled.columns if c.startswith("feat_")]
    if not feat_cols:
        return

    means = df_labeled.groupby("class")[feat_cols].mean(numeric_only=True)
    if 0 not in means.index or 1 not in means.index:
        return

    diff_series = (means.loc[1] - means.loc[0]).abs()
    diff_np = diff_series.to_numpy(dtype=float, copy=False)
    order = np.argsort(-diff_np)[:top_k]  # descending
    diff = diff_series.iloc[order]

    names = diff.index.tolist()
    vals = _to_np_1d(diff, dtype=float)

    plt.figure(figsize=(10, 5))
    plt.bar(names, vals)
    plt.title(f"Top-{top_k} features by |mean(illicit) - mean(licit)|")
    plt.xlabel("feature")
    plt.ylabel("absolute mean difference")
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "top_feature_mean_diff.png", dpi=200)
    plt.close()


def save_model_comparison_bar(model_to_f1: dict[str, float], out_dir: Path) -> None:
    _ensure_out(out_dir)
    names = list(model_to_f1.keys())
    vals = np.asarray([float(model_to_f1[k]) for k in names], dtype=float)

    plt.figure()
    plt.bar(names, vals)
    plt.title("Illicit F1 by model")
    plt.xlabel("model")
    plt.ylabel("F1 (illicit)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison_f1.png", dpi=200)
    plt.close()


# --- Phase05 model-eval visualizations ---

def save_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    out_dir: Path,
    *,
    fname: str = "pr_curve.png",
    title: str = "Precision-Recall curve",
) -> None:
    """
    precision, recall are typically from sklearn.metrics.precision_recall_curve.
    """
    _ensure_out(out_dir)
    p = _to_np_1d(precision, dtype=float)
    r = _to_np_1d(recall, dtype=float)
    if p.size == 0 or r.size == 0:
        return

    plt.figure()
    plt.plot(r, p)
    plt.title(title)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=200)
    plt.close()


def save_threshold_sweep(
    thresholds: np.ndarray,
    f1_illicit: np.ndarray,
    precision_illicit: np.ndarray,
    recall_illicit: np.ndarray,
    out_dir: Path,
    *,
    fname: str = "threshold_sweep.png",
    title: str = "Threshold sweep (val)",
) -> None:
    """
    Plot F1/precision/recall vs threshold.
    thresholds should be 1D floats in [0,1].
    """
    _ensure_out(out_dir)
    t = _to_np_1d(thresholds, dtype=float)
    f1 = _to_np_1d(f1_illicit, dtype=float)
    p = _to_np_1d(precision_illicit, dtype=float)
    r = _to_np_1d(recall_illicit, dtype=float)
    if t.size == 0:
        return

    # guard shape mismatches
    n = min(t.size, f1.size, p.size, r.size)
    t, f1, p, r = t[:n], f1[:n], p[:n], r[:n]

    plt.figure()
    plt.plot(t, f1, label="F1(illicit)")
    plt.plot(t, p, label="Precision(illicit)")
    plt.plot(t, r, label="Recall(illicit)")
    plt.title(title)
    plt.xlabel("threshold")
    plt.ylabel("score")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=200)
    plt.close()


def save_calibration_curve(
    prob_true: np.ndarray,
    prob_pred: np.ndarray,
    out_dir: Path,
    *,
    fname: str = "calibration_curve.png",
    title: str = "Calibration curve",
) -> None:
    """
    prob_true, prob_pred are typically from sklearn.calibration.calibration_curve.
    """
    _ensure_out(out_dir)
    pt = _to_np_1d(prob_true, dtype=float)
    pp = _to_np_1d(prob_pred, dtype=float)
    if pt.size == 0 or pp.size == 0:
        return

    plt.figure()
    plt.plot(pp, pt, marker="o")
    # Perfect calibration diagonal
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--")
    plt.title(title)
    plt.xlabel("mean predicted probability")
    plt.ylabel("fraction of positives")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=200)
    plt.close()


def save_metric_over_time(
    time_steps: np.ndarray,
    values: np.ndarray,
    out_dir: Path,
    *,
    fname: str = "metric_over_time.png",
    title: str = "Metric over time",
    ylabel: str = "value",
) -> None:
    """
    Generic: plot any scalar metric over time_step (e.g., illicit F1, recall, precision).
    """
    _ensure_out(out_dir)
    x = _to_np_1d(time_steps, dtype=int)
    y = _to_np_1d(values, dtype=float)
    if x.size == 0 or y.size == 0:
        return

    n = min(x.size, y.size)
    x, y = x[:n], y[:n]

    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("time_step")
    plt.ylabel(ylabel)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=200)
    plt.close()


def save_confusion_matrix_heatmap(
    cm: np.ndarray,
    out_dir: Path,
    *,
    fname: str = "confusion_matrix.png",
    title: str = "Confusion matrix",
    labels: tuple[str, str] = ("licit (0)", "illicit (1)"),
    normalize: bool = False,
) -> None:
    """
    cm is 2x2 with rows=true [0,1] and cols=pred [0,1].
    If normalize=True, rows sum to 1 (when possible).
    """
    _ensure_out(out_dir)
    a = np.asarray(cm, dtype=float)
    if a.shape != (2, 2):
        return

    if normalize:
        row_sum = a.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        a = a / row_sum

    plt.figure()
    plt.imshow(a, aspect="auto")
    plt.title(title + (" (normalized)" if normalize else ""))
    plt.xticks([0, 1], labels, rotation=15, ha="right")
    plt.yticks([0, 1], labels)
    plt.xlabel("predicted")
    plt.ylabel("true")

    # annotate cells
    for i in range(2):
        for j in range(2):
            text = f"{a[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            plt.text(j, i, text, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=200)
    plt.close()