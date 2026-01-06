from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


@dataclass
class GraphBundle:
    data: Data
    txid_to_idx: dict[int, int]
    feature_cols: list[str]
    scaler: Optional[StandardScaler]


def build_graph(
    df_labeled: pd.DataFrame,
    edges_df: pd.DataFrame,
    feature_cols: list[str],
    train_steps: np.ndarray,
    test_steps: np.ndarray,
    val_ratio_within_train: float = 0.10,
    normalize: bool = True,
) -> GraphBundle:
    """
    Single global graph + temporal masks.

    - Nodes = df_labeled rows (labeled tx only)
    - Edges = only between labeled nodes
    - Masks = by time_step (train/test), val is last slice of train time steps
    - Normalization (optional) fits scaler on TRAIN NODES ONLY (no leakage)
    """

    # Node indexing
    tx_ids = df_labeled["txId"].to_numpy()
    txid_to_idx = {int(tx): i for i, tx in enumerate(tx_ids)}

    # Temporal masks
    time_steps = df_labeled["time_step"].to_numpy()
    train_mask_np = np.isin(time_steps, train_steps)
    test_mask_np = np.isin(time_steps, test_steps)

    unique_train_steps = np.unique(train_steps)
    n_val_steps = max(1, int(len(unique_train_steps) * val_ratio_within_train))
    val_steps = unique_train_steps[-n_val_steps:]

    val_mask_np = np.isin(time_steps, val_steps)
    train_mask_np = train_mask_np & (~val_mask_np)

    # Node features
    X = df_labeled[feature_cols].to_numpy(dtype=np.float32)

    scaler: Optional[StandardScaler] = None
    if normalize:
        scaler = StandardScaler()
        X_train = X[train_mask_np]
        X[train_mask_np] = scaler.fit_transform(X_train).astype(np.float32)
        # transform non-train using same scaler
        X[~train_mask_np] = scaler.transform(X[~train_mask_np]).astype(np.float32)

    x = torch.tensor(X, dtype=torch.float32)

    # Labels: illicit=1 -> 1, licit=2 -> 0
    y = torch.tensor((df_labeled["class"].to_numpy() == 1).astype(np.int64), dtype=torch.long)

    # Edges (filter to nodes we kept)
    src = edges_df["src"].map(txid_to_idx)
    dst = edges_df["dst"].map(txid_to_idx)
    mask = src.notna() & dst.notna()

    edge_index = torch.tensor(
        np.vstack([src[mask].astype(int), dst[mask].astype(int)]),
        dtype=torch.long,
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=torch.tensor(train_mask_np, dtype=torch.bool),
        val_mask=torch.tensor(val_mask_np, dtype=torch.bool),
        test_mask=torch.tensor(test_mask_np, dtype=torch.bool),
    )

    return GraphBundle(
        data=data,
        txid_to_idx=txid_to_idx,
        feature_cols=feature_cols,
        scaler=scaler,
    )
