"""Prediction interfaces and a lazy PyTorch-Geometric adapter."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol

import numpy as np

from .community import Edge, canonical_edge


class GraphPredictor(Protocol):
    def __call__(self, graph: Any, removed_edges: Iterable[Edge] = ()) -> np.ndarray:
        """Return one-dimensional, pre-softmax class logits."""


def class_margin(logits: np.ndarray, target: int) -> float:
    logits = np.asarray(logits, dtype=float).reshape(-1)
    if logits.size < 2:
        raise ValueError("at least two class logits are required")
    if target < 0 or target >= logits.size:
        raise IndexError("target class is outside logits")
    alternatives = np.delete(logits, target)
    return float(logits[target] - alternatives.max())


class PyGGraphPredictor:
    """Wrap a graph-classification model taking ``(x, edge_index, batch)``.

    Edge deletion preserves all nodes and node features. This is important:
    community intervention removes edges, not nodes, so class logits remain
    comparable to the original graph prediction.
    """

    def __init__(self, model: Any, device: Any, forward: Callable[..., Any] | None = None):
        import torch

        self.torch = torch
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.forward = forward

    def __call__(self, graph: Any, removed_edges: Iterable[Edge] = ()) -> np.ndarray:
        torch = self.torch
        data = graph.clone()
        removed = {canonical_edge(*edge) for edge in removed_edges}
        if removed:
            edge_pairs = data.edge_index.detach().cpu().t().tolist()
            keep = [canonical_edge(u, v) not in removed for u, v in edge_pairs]
            keep_mask = torch.tensor(keep, dtype=torch.bool, device=data.edge_index.device)
            data.edge_index = data.edge_index[:, keep_mask]

        data = data.to(self.device)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=self.device)
        with torch.no_grad():
            if self.forward is None:
                output = self.model(data.x, data.edge_index, batch)
            else:
                output = self.forward(self.model, data, batch)
        return output.detach().cpu().numpy().reshape(-1)
