"""Server adapter for a trained BA2Motifs graph classifier."""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.datasets import BA2MotifDataset

from geco_cf.predictors import PyGGraphPredictor

from .ba2motifs_experiment import BA2MotifGCN, load_graphs, load_split


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

    dataset = BA2MotifDataset(root=str(data_root))
    split = load_split(split_path)
    evaluation_split = config.get("evaluation_split", "test")
    if evaluation_split not in split:
        raise ValueError(f"unsupported evaluation_split: {evaluation_split}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_kwargs = checkpoint.get("model_kwargs")
    if not isinstance(model_kwargs, dict):
        raise ValueError("checkpoint must contain model_kwargs")
    model = BA2MotifGCN(**model_kwargs)
    model.load_state_dict(checkpoint["model_state"])
    graphs = load_graphs(dataset, split[evaluation_split])
    return graphs, PyGGraphPredictor(model, device)
