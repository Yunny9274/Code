"""Model and data helpers for the Mutagenicity Stage-1 pilot."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GINConv, global_add_pool


def _gin_mlp(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.ReLU(),
        nn.Linear(out_channels, out_channels),
    )


class MutagenicityGIN(nn.Module):
    """Three-layer GIN graph classifier with sum pooling for molecular graphs."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv1 = GINConv(_gin_mlp(in_channels, hidden_channels))
        self.conv2 = GINConv(_gin_mlp(hidden_channels, hidden_channels))
        self.conv3 = GINConv(_gin_mlp(hidden_channels, hidden_channels))
        self.norm1 = BatchNorm(hidden_channels)
        self.norm2 = BatchNorm(hidden_channels)
        self.norm3 = BatchNorm(hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.norm1(F.relu(self.conv1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.norm2(F.relu(self.conv2(h, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.norm3(F.relu(self.conv3(h, edge_index)))
        return self.classifier(global_add_pool(h, batch))
