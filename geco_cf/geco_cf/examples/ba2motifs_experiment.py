"""Reproducible GCN and data helpers for the BA2Motifs pilot."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_add_pool, global_max_pool


class BA2MotifGCN(nn.Module):
    """Three-layer BA2Motifs GCN with max-plus-sum graph readout.

    This follows the original PGExplainer graph-classification readout. Max
    pooling preserves a motif activation while sum pooling retains its extent;
    mean pooling can dilute the small five-node motif in a 25-node graph.
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        # Match the original PGExplainer BA-2motif implementation: raw
        # adjacency aggregation preserves degree/motif signal when features
        # are constant. PyG's normalized self-loop default would wash it out.
        self.conv1 = GCNConv(
            in_channels, hidden_channels, normalize=False, add_self_loops=False
        )
        self.conv2 = GCNConv(
            hidden_channels, hidden_channels, normalize=False, add_self_loops=False
        )
        self.conv3 = GCNConv(
            hidden_channels, hidden_channels, normalize=False, add_self_loops=False
        )
        self.classifier = nn.Linear(hidden_channels * 2, num_classes)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.conv1(x, edge_index))
        h2 = torch.relu(self.conv2(F.dropout(h1, p=self.dropout, training=self.training), edge_index))
        h3 = torch.relu(self.conv3(F.dropout(h2, p=self.dropout, training=self.training), edge_index))
        pooled = torch.cat(
            [global_max_pool(h3, batch), global_add_pool(h3, batch)],
            dim=-1,
        )
        return self.classifier(pooled)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def stratified_split(labels: torch.Tensor, seed: int, train_fraction: float, val_fraction: float) -> dict[str, list[int]]:
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction and val_fraction must leave a non-empty test split")
    rng = np.random.default_rng(seed)
    labels_np = labels.detach().cpu().numpy().reshape(-1)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for label in sorted(np.unique(labels_np).tolist()):
        class_indices = np.flatnonzero(labels_np == label)
        rng.shuffle(class_indices)
        n_train = int(len(class_indices) * train_fraction)
        n_val = int(len(class_indices) * val_fraction)
        train.extend(int(i) for i in class_indices[:n_train])
        val.extend(int(i) for i in class_indices[n_train : n_train + n_val])
        test.extend(int(i) for i in class_indices[n_train + n_val :])
    for values in (train, val, test):
        values.sort()
    return {"train": train, "val": val, "test": test}


def load_graphs(dataset: Any, indices: list[int]) -> list[Data]:
    graphs: list[Data] = []
    for index in indices:
        graph = dataset[int(index)].clone()
        graph.graph_id = int(index)
        graphs.append(graph)
    return graphs


def save_split(path: str | Path, split: dict[str, list[int]], metadata: dict[str, Any]) -> None:
    payload = {"split": split, "metadata": metadata}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split(path: str | Path) -> dict[str, list[int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    split = payload.get("split", payload)
    required = {"train", "val", "test"}
    if set(split) != required:
        raise ValueError(f"split file must contain exactly {sorted(required)}")
    return {key: [int(index) for index in split[key]] for key in required}
