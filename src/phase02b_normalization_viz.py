# src/phase02b_normalization_viz.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import ExperimentConfig, Paths
from src.data import get_feature_cols
from src.repro import set_seed
from src.runlog import write_run_manifest


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _json_default(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    tolist = getattr(o, "tolist", None)
    if callable(tolist):
        return tolist()
    return str(o)


def _load_labeled(paths: Paths) -> pd.DataFrame:
    p = paths.processed_dir / "elliptic_labeled.parquet"
    c = paths.processed_dir / "elliptic_labeled.csv"
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError("Missing labeled dataset. Run: python -m src.main --phase 1")


def _temporal_train_test_steps(df_labeled: pd.DataFrame, train_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    steps = np.sort(df_labeled["time_step"].unique().astype(int))
    n_train = int(np.floor(len(steps) * float(train_ratio)))
    n_train = max(1, min(n_train, len(steps) - 1))
    return steps[:n_train], steps[n_train:]


def to_binary_labels(y: pd.Series) -> np.ndarray:
    y_int = y.to_numpy(dtype=np.int64)
    uniq = np.unique(y_int)
    if set(uniq.tolist()) <= {0, 1}:
        return y_int
    if set(uniq.tolist()) <= {1, 2}:
        # 1=illicit -> 1, 2=licit -> 0
        return (y_int == 1).astype(np.int64)
    raise ValueError(f"Unexpected class values: {uniq.tolist()}")



def _pick_feature_indices(n_features: int) -> List[int]:
    if n_features <= 0:
        return []
    k = min(16, n_features)
    idx = np.linspace(0, n_features - 1, num=k, dtype=int)
    return idx.tolist()


def _save_feature_histograms(
    X: np.ndarray,
    Xn: np.ndarray,
    feat_cols: List[str],
    out_dir: Path,
    *,
    prefix: str,
    bins: int = 80,
    post_xlim: float = 6.0,
) -> Dict[str, Any]:
    """
    Two figures (16 subplots each):
      - pre: raw histograms (auto x-range per subplot)
      - post: standardized histograms with fixed xlim and clipping to reduce outlier distortion
    """
    import matplotlib.pyplot as plt

    n = X.shape[1]
    sel = _pick_feature_indices(n)
    if not sel:
        return {"selected_features": []}

    names = [feat_cols[i] for i in sel]

    # Pre
    plt.figure(figsize=(12, 8))
    for i, fi in enumerate(sel):
        plt.subplot(4, 4, i + 1)
        plt.hist(X[:, fi], bins=bins, density=True)
        plt.title(names[i], fontsize=8)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
    plt.tight_layout()
    pre_path = out_dir / f"{prefix}_feature_hist_pre.png"
    plt.savefig(pre_path, dpi=200)
    plt.close()

    # Post (clip + fixed xlim makes it *look* standardized even with outliers)
    plt.figure(figsize=(12, 8))
    for i, fi in enumerate(sel):
        plt.subplot(4, 4, i + 1)
        z = np.clip(Xn[:, fi], -post_xlim, post_xlim)
        plt.hist(z, bins=bins, density=True)
        plt.xlim(-post_xlim, post_xlim)
        # reference lines
        plt.axvline(0.0, linewidth=1)
        plt.axvline(-1.0, linewidth=1)
        plt.axvline(1.0, linewidth=1)
        plt.title(names[i], fontsize=8)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
    plt.tight_layout()
    post_path = out_dir / f"{prefix}_feature_hist_post.png"
    plt.savefig(post_path, dpi=200)
    plt.close()

    return {
        "selected_feature_indices": sel,
        "selected_feature_names": names,
        "hist_pre": str(pre_path),
        "hist_post": str(post_path),
        "post_xlim": float(post_xlim),
    }


def _save_pca_scatter(
    X: np.ndarray,
    y_bin: np.ndarray,
    out_dir: Path,
    *,
    prefix: str,
    title_suffix: str,
    max_points: int = 20000,
) -> str:
    import matplotlib.pyplot as plt

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y_bin, dtype=np.int64)

    n = X.shape[0]
    if n == 0:
        return ""

    if n > max_points:
        idx = np.linspace(0, n - 1, num=max_points, dtype=int)
        Xp = X[idx]
        yp = y[idx]
    else:
        Xp = X
        yp = y

    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xp)

    plt.figure(figsize=(8, 6))
    for cls in [0, 1]:
        m = (yp == cls)
        if m.any():
            plt.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.6, label=f"class={cls}")
    plt.legend(loc="best")
    plt.title(f"PCA(2) {title_suffix}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

    out_path = out_dir / f"{prefix}_pca_{title_suffix.lower().replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return str(out_path)


def _save_mean_std_diagnostics(Xn: np.ndarray, out_dir: Path, *, prefix: str) -> Dict[str, Any]:
    import matplotlib.pyplot as plt
    Xn = np.asarray(Xn, dtype=np.float64)

    mu = Xn.mean(axis=0)
    sd = Xn.std(axis=0)  # ddof=0 (matches StandardScaler)

    # guard non-finite
    mu = mu[np.isfinite(mu)]
    sd = sd[np.isfinite(sd)]

    def _safe_hist(data: np.ndarray, *, bins: int = 80):
        data = np.asarray(data, dtype=np.float64)
        if data.size == 0:
            return dict(min=None, max=None, ptp=None)
        lo = float(np.min(data))
        hi = float(np.max(data))
        ptp = hi - lo

        # If range collapses, expand slightly and reduce bins
        if not np.isfinite(ptp) or ptp <= 0.0:
            eps = 1e-6 if lo == 0.0 else abs(lo) * 1e-6
            lo, hi = lo - eps, hi + eps
            b = 10
        else:
            # If range is extremely tiny, force explicit edges
            b = min(bins, max(10, int(np.sqrt(data.size))))
        edges_arr = np.linspace(lo, hi, b + 1, dtype=float)
        edges: Sequence[float] = edges_arr.tolist()
        plt.hist(data, bins=edges, density=True)

        return dict(min=float(np.min(data)), max=float(np.max(data)), ptp=float(ptp))

    # means histogram
    plt.figure(figsize=(8, 4))
    mstats = _safe_hist(mu, bins=80)
    plt.title(f"{prefix}: per-feature means after scaling")
    plt.xlabel("mean")
    plt.ylabel("density")
    mean_path = out_dir / f"{prefix}_means_hist.png"
    plt.tight_layout()
    plt.savefig(mean_path, dpi=200)
    plt.close()

    # stds histogram
    plt.figure(figsize=(8, 4))
    sstats = _safe_hist(sd, bins=80)
    plt.title(f"{prefix}: per-feature stds after scaling")
    plt.xlabel("std")
    plt.ylabel("density")
    std_path = out_dir / f"{prefix}_stds_hist.png"
    plt.tight_layout()
    plt.savefig(std_path, dpi=200)
    plt.close()

    return {
        "means_hist": str(mean_path),
        "stds_hist": str(std_path),
        "means_stats": mstats,
        "stds_stats": sstats,
    }


def _summarize_scaler(scaler: StandardScaler, feat_cols: List[str]) -> Dict[str, Any]:
    mu = np.asarray(scaler.mean_, dtype=float)
    sd = np.asarray(scaler.scale_, dtype=float)
    return {
        "n_features": int(mu.size),
        "mean_abs_mean": float(np.mean(np.abs(mu))),
        "mean_std": float(np.mean(sd)),
        "min_std": float(np.min(sd)),
        "max_std": float(np.max(sd)),
        "top10_abs_mean_features": [
            {"feature": feat_cols[i], "abs_mean": float(abs(mu[i])), "mean": float(mu[i])}
            for i in np.argsort(-np.abs(mu))[:10].tolist()
        ],
        "bottom10_std_features": [
            {"feature": feat_cols[i], "std": float(sd[i])}
            for i in np.argsort(sd)[:10].tolist()
        ],
    }


def main() -> None:
    paths = Paths()
    cfg = ExperimentConfig()
    set_seed(cfg.seed, deterministic_torch=False)

    out_dir = paths.results_dir / "visualizations" / "normalizedData"
    _ensure_dir(out_dir)
    (paths.results_dir / "logs").mkdir(parents=True, exist_ok=True)

    write_run_manifest(
        paths.results_dir / "logs",
        phase="02b",
        cfg=cfg,
        data_files=[
            paths.processed_dir / "elliptic_labeled.parquet",
            paths.processed_dir / "elliptic_labeled.csv",
        ],
        extra={"note": "phase02b: train-only StandardScaler; improved viz (label mapping + clipped post hist + mean/std diagnostics)"},
    )

    df = _load_labeled(paths).sort_values(["time_step", "txId"]).reset_index(drop=True)

    feature_mode = "AF"
    feat_cols = get_feature_cols(df, feature_mode)  # type: ignore[arg-type]

    train_steps, test_steps = _temporal_train_test_steps(df, cfg.train_ratio)
    df_train = df[df["time_step"].isin(train_steps)].copy()
    df_test = df[df["time_step"].isin(test_steps)].copy()

    X_train = df_train[feat_cols].to_numpy(dtype=np.float32)
    y_train = to_binary_labels(df_train["class"])
    X_test = df_test[feat_cols].to_numpy(dtype=np.float32)
    y_test  = to_binary_labels(df_test["class"])

    if X_train.size == 0 or X_test.size == 0:
        raise RuntimeError("Empty train/test after temporal split; check processed dataset.")

    scaler = StandardScaler()
    X_train_n = scaler.fit_transform(X_train)
    X_test_n = scaler.transform(X_test)

    # Visuals
    hist_info = _save_feature_histograms(
        X_train, X_train_n, feat_cols, out_dir, prefix="AF_train", bins=80, post_xlim=6.0
    )
    diag_info = _save_mean_std_diagnostics(X_train_n, out_dir, prefix="AF")

    pca_pre = _save_pca_scatter(X_train, y_train, out_dir, prefix="AF_train", title_suffix="pre_norm")
    pca_post = _save_pca_scatter(X_train_n, y_train, out_dir, prefix="AF_train", title_suffix="post_norm")

    # Quantitative checks
    train_mean = X_train_n.mean(axis=0)
    train_std = X_train_n.std(axis=0)
    test_mean = X_test_n.mean(axis=0)
    test_std = X_test_n.std(axis=0)

    report: Dict[str, Any] = {
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "feature_mode": feature_mode,
        "split": {
            "train_ratio": float(cfg.train_ratio),
            "train_steps": train_steps.tolist(),
            "test_steps": test_steps.tolist(),
            "train_rows": int(df_train.shape[0]),
            "test_rows": int(df_test.shape[0]),
        },
        "labels": {
            "train_unique_binary": sorted(np.unique(y_train).tolist()),
            "test_unique_binary": sorted(np.unique(y_test).tolist()),
            "note": "Elliptic mapping: raw 1->illicit(1), raw 2->licit(0)",
        },
        "scaler_fit": "train_only",
        "scaler_summary": _summarize_scaler(scaler, feat_cols),
        "sanity_train_after_norm": {
            "mean_abs_mean_over_features": float(np.mean(np.abs(train_mean))),
            "mean_std_over_features": float(np.mean(train_std)),
            "min_std_over_features": float(np.min(train_std)),
            "max_std_over_features": float(np.max(train_std)),
        },
        "sanity_test_after_norm": {
            "mean_abs_mean_over_features": float(np.mean(np.abs(test_mean))),
            "mean_std_over_features": float(np.mean(test_std)),
            "min_std_over_features": float(np.min(test_std)),
            "max_std_over_features": float(np.max(test_std)),
            "note": "Test will NOT be exactly 0-mean/1-std because scaler is fit on train only.",
        },
        "artifacts": {
            **hist_info,
            **diag_info,
            "pca_pre": pca_pre,
            "pca_post": pca_post,
        },
    }

    out_path = out_dir / "normalized_stats_AF.json"
    out_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")


if __name__ == "__main__":
    main()
