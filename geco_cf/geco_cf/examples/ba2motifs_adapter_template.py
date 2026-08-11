"""Copy and complete this adapter once the reproduced checkpoint is ready."""

from __future__ import annotations

from geco_cf.predictors import PyGGraphPredictor


def build_experiment(config: dict):
    import torch

    # Replace these two imports with the dataset and model from your successful
    # BA2Motifs reproduction. Keep its exact split and architecture.
    raise NotImplementedError(
        "Load the reproduced BA2Motifs dataset/model here, then return "
        "(test_dataset, PyGGraphPredictor(model, device))."
    )

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = ...
    # model.load_state_dict(torch.load("checkpoint.pt", map_location=device))
    # test_dataset = ...
    # return test_dataset, PyGGraphPredictor(model, device)
