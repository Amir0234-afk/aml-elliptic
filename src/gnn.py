# src/gnn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Protocol, Tuple, runtime_checkable

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x: Iterable[int], **kwargs: Any) -> Iterable[int]:  # type: ignore
        return x


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 2)
        self.dropout = float(dropout)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encode(x, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, edge_index)
        return out


class SkipGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 2)
        self.skip = nn.Linear(in_dim, 2, bias=True)
        self.dropout = float(dropout)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encode(x, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, edge_index)
        out = out + self.skip(x)
        return out


@runtime_checkable
class _TqdmLike(Protocol):
    def set_postfix(self, ordered_dict: Any = ..., refresh: bool = ...) -> None: ...


@dataclass
class GNNTrainResult:
    best_state_dict: dict[str, torch.Tensor]
    best_monitor_f1: float
    best_epoch: int


def _f1_illicit_from_logits(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    # illicit label is 1, licit label is 0
    y_pred = logits.argmax(dim=1)
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return float(f1)


def train_gnn(
    model: nn.Module,
    data,  # torch_geometric.data.Data
    train_mask: torch.Tensor,
    monitor_mask: torch.Tensor,
    class_weights: Tuple[float, float],  # (w0 for licit, w1 for illicit)
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    device: str = "cpu",
    show_progress: bool = True,
) -> GNNTrainResult:
    model = model.to(device)
    data = data.to(device)

    # Safety: ensure masks don't include unlabeled nodes (-1)
    y_train = data.y[train_mask]
    y_mon = data.y[monitor_mask]
    if (y_train < 0).any().item():
        raise ValueError("train_mask contains unlabeled nodes (class=-1). Fix mask construction.")
    if (y_mon < 0).any().item():
        raise ValueError("monitor_mask contains unlabeled nodes (class=-1). Fix mask construction.")

    w = torch.tensor(class_weights, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_f1 = -1.0
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_epoch = 0
    bad = 0

    iterator: Iterable[int] = range(1, int(epochs) + 1)
    if show_progress:
        iterator = tqdm(iterator, desc=f"GNN epochs ({device})", leave=False)

    for epoch in iterator:
        model.train()
        opt.zero_grad()

        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask], weight=w)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            m_logits = model(data.x, data.edge_index)[monitor_mask]
            m_y = data.y[monitor_mask]
            monitor_f1 = _f1_illicit_from_logits(m_logits, m_y)

        if show_progress and isinstance(iterator, _TqdmLike):
            iterator.set_postfix({"monitor_f1": f"{monitor_f1:.4f}", "loss": f"{loss.item():.4f}"})

        if monitor_f1 > best_f1 + 1e-6:
            best_f1 = monitor_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if int(patience) > 0 and bad >= int(patience):
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return GNNTrainResult(best_state_dict=best_state, best_monitor_f1=float(best_f1), best_epoch=int(best_epoch))
