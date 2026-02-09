# src/graph_data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class GraphBundle:
    data: Data
    feature_cols: list[str]
    train_steps: np.ndarray
    val_steps: np.ndarray
    test_steps: np.ndarray


def build_graph(
    df_labeled: pd.DataFrame,              # now allowed to be FULL dataset (contains class=-1)
    edges_df: pd.DataFrame,
    feature_cols: list[str],
    train_steps: np.ndarray,
    test_steps: np.ndarray,
    val_ratio_within_train: float = 0.1,
    normalize: bool = True,
    make_undirected: bool = True,
) -> GraphBundle:
    df = df_labeled.copy()
    
    # Deterministic node ordering
    df = df.sort_values("txId").reset_index(drop=True)
    if not df["txId"].is_unique:
        raise ValueError("build_graph: txId is not unique (node mapping would be ambiguous).")


    # Ensure required columns exist
    req = {"txId", "time_step", "class", *feature_cols}
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"build_graph: missing columns: {missing}")

    # Node ordering / index map
    tx_ids = df["txId"].to_numpy(dtype=np.int64)
    id2idx = {int(t): i for i, t in enumerate(tx_ids)}

    # Edges (filter to nodes in df)
    src = edges_df["txId1"].to_numpy(dtype=np.int64)
    dst = edges_df["txId2"].to_numpy(dtype=np.int64)

    keep = np.isin(src, tx_ids) & np.isin(dst, tx_ids)
    src = src[keep]
    dst = dst[keep]

    row = np.fromiter((id2idx[int(s)] for s in src), dtype=np.int64, count=len(src))
    col = np.fromiter((id2idx[int(d)] for d in dst), dtype=np.int64, count=len(dst))

    edge_index = torch.tensor(np.vstack([row, col]), dtype=torch.long)
    if make_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Labels (full graph includes unlabeled: -1)
    y_np = df["class"].to_numpy(dtype=np.int64)
    y = torch.tensor(y_np, dtype=torch.long)

    # Temporal splits
    train_steps = np.asarray(train_steps, dtype=np.int64)
    test_steps = np.asarray(test_steps, dtype=np.int64)

    unique_train = np.sort(np.unique(train_steps))
    if len(unique_train) < 2:
        # degenerate; put all in train, no val
        val_steps = unique_train.copy()
        train_only_steps = unique_train.copy()
    else:
        n_val = int(np.floor(len(unique_train) * val_ratio_within_train))
        n_val = max(1, min(n_val, len(unique_train) - 1))
        val_steps = unique_train[-n_val:]
        train_only_steps = unique_train[:-n_val]

    ts = df["time_step"].to_numpy(dtype=np.int64)
    is_labeled = (y_np != -1)

    train_mask = torch.tensor(np.isin(ts, train_only_steps) & is_labeled, dtype=torch.bool)
    val_mask = torch.tensor(np.isin(ts, val_steps) & is_labeled, dtype=torch.bool)
    test_mask = torch.tensor(np.isin(ts, test_steps) & is_labeled, dtype=torch.bool)

    # Features
    X = df[feature_cols].to_numpy(dtype=np.float32)
    # Fill NaNs defensively
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if normalize:
        # Fit normalization on TRAIN LABELED NODES ONLY (no leakage)
        idx = train_mask.numpy()
        if idx.sum() == 0:
            raise RuntimeError("No labeled train nodes in train_mask; check split / labels.")
        mu = X[idx].mean(axis=0)
        sd = X[idx].std(axis=0) + 1e-8
        X = (X - mu) / sd

    x = torch.tensor(X, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    return GraphBundle(
        data=data,
        feature_cols=feature_cols,
        train_steps=train_only_steps,
        val_steps=val_steps,
        test_steps=test_steps,
    )
