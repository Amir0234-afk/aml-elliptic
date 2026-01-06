from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import StandardScaler

ELLIPTIC_FEATURES_CANDIDATES = [
    "elliptic_txs_features.csv",
    "txs_features.csv",
    "features.csv",
]
ELLIPTIC_EDGES_CANDIDATES = [
    "elliptic_txs_edgelist.csv",
    "txs_edgelist.csv",
    "edgelist.csv",
    "edges.csv",
]
ELLIPTIC_CLASSES_CANDIDATES = [
    "elliptic_txs_classes.csv",
    "txs_classes.csv",
    "classes.csv",
    "labels.csv",
]


@dataclass
class LoadedElliptic:
    features: pd.DataFrame  # txId, f0..f165, time_step
    edges: pd.DataFrame     # src, dst
    classes: pd.DataFrame   # txId, class


def _find_file(data_dir: Path, candidates: list[str]) -> Path:
    for name in candidates:
        p = data_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find any of: {candidates} in {data_dir.resolve()}")


def load_features(data_dir: Path) -> pd.DataFrame:
    f_path = _find_file(data_dir, ELLIPTIC_FEATURES_CANDIDATES)

    # Elliptic features: 167 columns total = txId + 166 features
    # Some exports introduce an extra leading column; drop it defensively.
    features = pd.read_csv(f_path, header=None, index_col=False)

    if features.shape[1] == 168:
        features = features.iloc[:, 1:]

    n_cols = features.shape[1]
    if n_cols != 167:
        raise ValueError(f"Features file has {n_cols} cols; expected exactly 167 (txId + 166 features).")

    col_names = ["txId"] + [f"f{i}" for i in range(n_cols - 1)]
    features.columns = col_names

    features["txId"] = pd.to_numeric(features["txId"], errors="raise").astype(int)
    # time step is encoded in f0
    features["time_step"] = pd.to_numeric(features["f0"], errors="raise").astype(int)

    return features


def load_edges(data_dir: Path) -> pd.DataFrame:
    e_path = _find_file(data_dir, ELLIPTIC_EDGES_CANDIDATES)

    edges = pd.read_csv(e_path)
    if edges.shape[1] < 2:
        raise ValueError("Edges file must have at least 2 columns (src, dst).")

    edges = edges.iloc[:, :2].copy()
    edges.columns = ["src", "dst"]

    edges["src"] = pd.to_numeric(edges["src"], errors="raise").astype(int)
    edges["dst"] = pd.to_numeric(edges["dst"], errors="raise").astype(int)
    return edges


def load_classes(data_dir: Path) -> pd.DataFrame:
    c_path = _find_file(data_dir, ELLIPTIC_CLASSES_CANDIDATES)

    classes = pd.read_csv(c_path)
    if classes.shape[1] < 2:
        raise ValueError("Classes file must have at least 2 columns (txId, class).")

    classes = classes.iloc[:, :2].copy()
    classes.columns = ["txId", "class"]
    classes["txId"] = pd.to_numeric(classes["txId"], errors="raise").astype(int)
    return classes


def load_elliptic(data_dir: Path) -> LoadedElliptic:
    return LoadedElliptic(
        features=load_features(data_dir),
        edges=load_edges(data_dir),
        classes=load_classes(data_dir),
    )


def labeled_only(df_feat: pd.DataFrame, df_cls: pd.DataFrame) -> pd.DataFrame:
    merged = df_feat.merge(df_cls, on="txId", how="left")

    # class can be "1","2","unknown" => coerce numeric; unknown -> NaN
    merged["class"] = pd.to_numeric(merged["class"], errors="coerce")
    merged = merged.loc[merged["class"].isin([1, 2])].copy()
    merged["class"] = merged["class"].astype(int)

    return merged


def get_feature_cols(df: pd.DataFrame, feature_mode: str) -> List[str]:
    """
    feature_mode:
      LF: f1..f94 (94 cols)
      AF: f1..f165 (166 cols)
    Excludes f0 because it encodes time_step.
    """
    if feature_mode not in ("LF", "AF"):
        raise ValueError("feature_mode must be 'LF' or 'AF'")

    f_cols = [c for c in df.columns if c.startswith("f")]
    f_cols = sorted(f_cols, key=lambda s: int(s[1:]))
    feat_cols = [c for c in f_cols if c != "f0"]

    if feature_mode == "LF":
        return feat_cols[:94]
    return feat_cols[:166]


def temporal_split_time_steps(time_steps: npt.ArrayLike, train_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    ts = np.asarray(time_steps)
    unique_steps = np.unique(ts)
    unique_steps.sort()

    if unique_steps.size < 2:
        raise ValueError("Need at least 2 unique time steps for a temporal train/test split.")

    cut = int(np.floor(train_ratio * unique_steps.size))
    cut = max(1, min(cut, unique_steps.size - 1))

    return unique_steps[:cut], unique_steps[cut:]


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    tx_train: np.ndarray
    tx_test: np.ndarray
    train_steps: np.ndarray
    test_steps: np.ndarray
    feature_cols: list[str]
    scaler: Optional[StandardScaler]


def make_tabular_split(
    labeled_df: pd.DataFrame,
    feature_mode: str,
    train_ratio: float,
    normalize: bool,
) -> SplitData:
    train_steps, test_steps = temporal_split_time_steps(labeled_df["time_step"].to_numpy(), train_ratio)

    df_train = labeled_df[labeled_df["time_step"].isin(train_steps)].copy()
    df_test = labeled_df[labeled_df["time_step"].isin(test_steps)].copy()

    feature_cols = get_feature_cols(labeled_df, feature_mode)

    X_train = df_train[feature_cols].to_numpy(dtype=np.float32)
    X_test = df_test[feature_cols].to_numpy(dtype=np.float32)

    # illicit=1 -> 1, licit=2 -> 0
    y_train = (df_train["class"].to_numpy() == 1).astype(np.int64)
    y_test = (df_test["class"].to_numpy() == 1).astype(np.int64)

    scaler: Optional[StandardScaler] = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

    return SplitData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        tx_train=df_train["txId"].to_numpy(),
        tx_test=df_test["txId"].to_numpy(),
        train_steps=train_steps,
        test_steps=test_steps,
        feature_cols=feature_cols,
        scaler=scaler,
    )


def class_weights_binary(y: np.ndarray) -> Dict[int, float]:
    n = int(y.shape[0])
    n_pos = int(y.sum())
    n_neg = n - n_pos
    w0 = n / max(1, 2 * n_neg)
    w1 = n / max(1, 2 * n_pos)
    return {0: float(w0), 1: float(w1)}
