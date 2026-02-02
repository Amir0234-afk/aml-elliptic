# src/data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, TypeAlias

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CSVHeader: TypeAlias = int | Sequence[int] | Literal["infer"] | None
FeatureMode = Literal["AF", "LF"]


@dataclass(frozen=True)
class EllipticLoaded:
    features: pd.DataFrame  # txId, time_step, feat_*
    edges: pd.DataFrame     # txId1, txId2
    classes: pd.DataFrame   # txId, class (1/2/unknown)


@dataclass(frozen=True)
class TabularSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    tx_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    tx_test: np.ndarray
    train_steps: np.ndarray
    test_steps: np.ndarray
    feature_cols: list[str]
    scaler: Optional[StandardScaler]


def read_csv_typed(path: Path, *, header: CSVHeader = "infer", **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, header=header, **kwargs)


def _read_csv_flexible(path: Path, header: CSVHeader = "infer") -> pd.DataFrame:
    return read_csv_typed(path, header=header)


def _coerce_int_series(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    return out


def _load_features(raw_dir: Path) -> pd.DataFrame:
    p = raw_dir / "elliptic_txs_features.csv"
    # Elliptic features file is typically headerless; read as header=None.
    df = _read_csv_flexible(p, header=None)

    if df.shape[1] < 3:
        # Fallback
        df = _read_csv_flexible(p, header="infer")

    n_feat = df.shape[1] - 2
    cols = ["txId", "time_step"] + [f"feat_{i}" for i in range(n_feat)]
    df = df.iloc[:, : (2 + n_feat)].copy()
    df.columns = cols

    df["txId"] = _coerce_int_series(df["txId"])
    df["time_step"] = _coerce_int_series(df["time_step"])

    # Drop any accidental header row / corrupt rows
    df = df.dropna(subset=["txId", "time_step"]).copy()
    df["txId"] = df["txId"].astype(np.int64)
    df["time_step"] = df["time_step"].astype(np.int64)

    # Features to float
    feat_cols = cols[2:]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Defensive NaN handling (should be rare in Elliptic)
    df[feat_cols] = df[feat_cols].fillna(0.0)

    # Deterministic ordering
    df = df.sort_values("txId").reset_index(drop=True)
    return df


def _load_edges(raw_dir: Path) -> pd.DataFrame:
    p = raw_dir / "elliptic_txs_edgelist.csv"
    # Robust: read as headerless, then coerce numeric and drop NaNs (handles header row if present).
    df = _read_csv_flexible(p, header=None)
    if df.shape[1] < 2:
        df = _read_csv_flexible(p, header="infer")

    df = df.iloc[:, :2].copy()
    df.columns = ["txId1", "txId2"]

    df["txId1"] = _coerce_int_series(df["txId1"])
    df["txId2"] = _coerce_int_series(df["txId2"])
    df = df.dropna(subset=["txId1", "txId2"]).copy()

    df["txId1"] = df["txId1"].astype(np.int64)
    df["txId2"] = df["txId2"].astype(np.int64)

    return df


def _load_classes(raw_dir: Path) -> pd.DataFrame:
    p = raw_dir / "elliptic_txs_classes.csv"

    # Robust: read headerless then coerce; if header exists, it will be kept as row 0 and dropped by coercion.
    df = _read_csv_flexible(p, header=None)
    if df.shape[1] < 2:
        df = _read_csv_flexible(p, header="infer")

    df = df.iloc[:, :2].copy()
    df.columns = ["txId", "class"]

    df["txId"] = _coerce_int_series(df["txId"])
    df = df.dropna(subset=["txId"]).copy()
    df["txId"] = df["txId"].astype(np.int64)
    df["class"] = df["class"].astype(str).str.strip()

    return df


def load_elliptic(raw_dir: Path) -> EllipticLoaded:
    feats = _load_features(raw_dir)
    edges = _load_edges(raw_dir)
    classes = _load_classes(raw_dir)
    return EllipticLoaded(features=feats, edges=edges, classes=classes)


def map_class_to_int(series: pd.Series) -> pd.Series:
    """
    Elliptic: '1' = illicit, '2' = licit, 'unknown' = unlabeled.
    Return: 1=illicit, 0=licit, -1=unknown.
    """
    s = series.astype(str).str.strip().str.lower()
    out = pd.Series(-1, index=s.index, dtype="int64")
    out.loc[s == "1"] = 1
    out.loc[s == "2"] = 0
    out.loc[s == "unknown"] = -1
    return out


def build_full_dataset(features: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    df = features.merge(classes[["txId", "class"]], on="txId", how="left")
    df["class_raw"] = df["class"].astype(str)
    df["class"] = map_class_to_int(df["class_raw"])
    df["class"] = df["class"].fillna(-1).astype(int)

    # Deterministic ordering
    df = df.sort_values("txId").reset_index(drop=True)
    return df


def labeled_only(features: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    df_full = build_full_dataset(features, classes)
    df_lab = df_full[df_full["class"] != -1].copy()
    df_lab["class"] = df_lab["class"].astype(int)
    return df_lab


def _sorted_feat_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("feat_")]
    if not cols:
        return []
    def key(c: str) -> int:
        try:
            return int(c.split("_", 1)[1])
        except Exception:
            return 10**12
    return sorted(cols, key=key)


def get_feature_cols(df: pd.DataFrame, feature_mode: FeatureMode) -> list[str]:
    feat_cols = _sorted_feat_cols(df)
    if not feat_cols:
        raise ValueError("No feature columns found (expected columns starting with 'feat_').")

    if feature_mode == "AF":
        return feat_cols

    # LF: first 94 local features (Elliptic convention).
    k = min(94, len(feat_cols))
    return feat_cols[:k]


def load_edges(raw_dir: Path) -> pd.DataFrame:
    return _load_edges(raw_dir)


def class_weights_binary(y: np.ndarray) -> dict[int, float]:
    y = np.asarray(y).astype(int)
    classes = set(np.unique(y).tolist())
    if classes != {0, 1}:
        raise ValueError(f"class_weights_binary expects y in {{0,1}} only. Got classes={classes}.")
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    n = n0 + n1
    w0 = n / (2 * n0) if n0 > 0 else 1.0
    w1 = n / (2 * n1) if n1 > 0 else 1.0
    return {0: float(w0), 1: float(w1)}


def make_tabular_split(
    df_labeled: pd.DataFrame,
    feature_mode: FeatureMode,
    train_ratio: float = 0.7,
    normalize: bool = True,
) -> TabularSplit:
    df = df_labeled.copy()

    # Require labeled-only for tabular
    if (df["class"] == -1).any():
        df = df[df["class"] != -1].copy()

    df["class"] = df["class"].astype(int)
    if not set(df["class"].unique().tolist()).issubset({0, 1}):
        raise ValueError("Tabular split requires class in {0,1}. Re-run preprocessing.")

    steps = np.sort(df["time_step"].unique())
    if len(steps) < 2:
        raise ValueError("Not enough unique timesteps for temporal split.")

    n_train_steps = int(np.floor(len(steps) * train_ratio))
    n_train_steps = max(1, min(n_train_steps, len(steps) - 1))
    train_steps = steps[:n_train_steps]
    test_steps = steps[n_train_steps:]

    df_train = df[df["time_step"].isin(train_steps)].copy()
    df_test = df[df["time_step"].isin(test_steps)].copy()

    feature_cols = get_feature_cols(df, feature_mode)

    X_train = df_train[feature_cols].to_numpy(dtype=np.float32)
    y_train = df_train["class"].to_numpy(dtype=np.int64)
    tx_train = df_train["txId"].to_numpy(dtype=np.int64)

    X_test = df_test[feature_cols].to_numpy(dtype=np.float32)
    y_test = df_test["class"].to_numpy(dtype=np.int64)
    tx_test = df_test["txId"].to_numpy(dtype=np.int64)

    scaler: Optional[StandardScaler] = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return TabularSplit(
        X_train=X_train,
        y_train=y_train,
        tx_train=tx_train,
        X_test=X_test,
        y_test=y_test,
        tx_test=tx_test,
        train_steps=train_steps,
        test_steps=test_steps,
        feature_cols=feature_cols,
        scaler=scaler,
    )
