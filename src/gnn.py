from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class SkipGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 2)
        self.skip = nn.Linear(in_dim, 2, bias=True)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, edge_index)
        out = out + self.skip(x)
        return out


@dataclass
class GNNTrainResult:
    best_state_dict: dict
    best_val_f1: float


def _f1_illicit_from_logits(logits: torch.Tensor, y_true: torch.Tensor) -> float:
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
    data,                     # torch_geometric.data.Data
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    class_weights: Tuple[float, float],
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    device: str = "cpu",
) -> GNNTrainResult:
    model = model.to(device)
    data = data.to(device)

    w = torch.tensor(class_weights, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_f1 = -1.0
    best_state = None
    bad = 0

    for _epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask], weight=w)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(data.x, data.edge_index)[val_mask]
            val_y = data.y[val_mask]
            val_f1 = _f1_illicit_from_logits(val_logits, val_y)

        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return GNNTrainResult(best_state_dict=best_state, best_val_f1=best_f1)
