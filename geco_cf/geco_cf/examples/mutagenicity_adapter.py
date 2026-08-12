"""Server adapter for the frozen Mutagenicity graph classifier."""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.datasets import TUDataset

from geco_cf.predictors import PyGGraphPredictor

from .ba2motifs_experiment import load_graphs, load_split
from .mutagenicity_experiment import MutagenicityGIN


def _path(config: dict, key: str) -> Path:
    value = config.get(key)
    if not value:
        raise ValueError(f"config is missing '{key}'")
    return Path(value)


def build_experiment(config: dict):
    data_root = _path(config, "data_root")
    checkpoint_path = _path(config, "checkpoint")
    split_path = _path(config, "split")
    device = torch.device(config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    dataset = TUDataset(root=str(data_root), name="Mutagenicity", use_node_attr=False)
    split = load_split(split_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_kwargs = checkpoint.get("model_kwargs")
    if not isinstance(model_kwargs, dict):
        raise ValueError("checkpoint must contain model_kwargs")
    model = MutagenicityGIN(**model_kwargs)
    model.load_state_dict(checkpoint["model_state"])
    return load_graphs(dataset, split["test"]), PyGGraphPredictor(model, device)
