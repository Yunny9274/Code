"""Smoke test: verify the C2Explainer port runs on BA-2Motifs locally.

Checks, in order:
  1. checkpoint loads into the paper's GCN definition
  2. the pretrained model is actually accurate (sanity)
  3. C2Explainer produces valid counterfactuals, and how long a single
     graph takes -- this is the number that decides the whole experiment budget.

Run with:  D:/Anaconda/envs/pytorch/python.exe run_smoke.py
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.explain import Explainer
from torch_geometric.nn import global_max_pool

from cf_explainer import C2Explainer
from cf_explainer.gcn_conv import GCNConv


class GCN(torch.nn.Module):
    """Exactly as defined in exp/exp1_3.ipynb (the paper's BA-2Motifs model)."""

    def __init__(self, nhid: int, nout: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(10, nhid, normalize=False)
        self.conv2 = GCNConv(nhid, nhid, normalize=False)
        self.conv3 = GCNConv(nhid, nout, normalize=False)
        self.lin = Linear(nout, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight)
        x = global_max_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


def main() -> None:
    device = torch.device("cpu")

    print("=" * 62)
    print("[1] loading checkpoint")
    model = GCN(nhid=20, nout=20, dropout=0).to(device)
    state = torch.load(
        REPO / "models" / "GCN_BA2Motifs_sd.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    OK: {len(state)} tensors, {n_params} parameters")

    print("[2] loading dataset")
    with open(REPO / "data" / "BA2motifs.pickle", "rb") as fh:
        dataset = pickle.load(fh)
    print(f"    type = {type(dataset).__module__}.{type(dataset).__name__}")
    # The authors pickled these on GPU, so the container carries a
    # ToDevice('cuda') transform. We want raw CPU graphs -- drop all transforms.
    if hasattr(dataset, "transform"):
        print(f"    dropping transform = {dataset.transform}")
        dataset.transform = None
    if hasattr(dataset, "pre_transform"):
        dataset.pre_transform = None

    graphs = list(dataset._data) if hasattr(dataset, "_data") else list(dataset)
    print(f"    {len(graphs)} graphs | node feat dim = {graphs[0].x.size(1)}")
    print(f"    labels = {sorted(set(int(d.y.item()) for d in graphs))}")

    gen = torch.Generator().manual_seed(42)
    order = torch.randperm(len(graphs), generator=gen).tolist()
    graphs = [graphs[i] for i in order]
    test_dataset = graphs[int(len(graphs) * 0.8):]
    print(f"    test split = {len(test_dataset)} graphs")

    print("[3] model accuracy on test split (sanity check the port is faithful)")
    correct = 0
    with torch.no_grad():
        for data in test_dataset:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            pred = model(data.x.to(device), data.edge_index.to(device), batch)
            correct += int(pred.argmax().item() == int(data.y.item()))
    print(f"    accuracy = {correct}/{len(test_dataset)} = {correct / len(test_dataset):.3f}")

    print("[4] running C2Explainer (epochs=100 for smoke, paper uses 1000)")
    algorithm = C2Explainer(epochs=100, lr=0.1, silent_mode=True, undirected=True)
    explainer = Explainer(
        model=model,
        algorithm=algorithm,
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="graph",
            return_type="raw",
        ),
    )

    n_try = 5
    found, timings = 0, []
    for i, data in enumerate(test_dataset[:n_try]):
        start = time.time()
        explanation = explainer(data.x.to(device), data.edge_index.to(device), batch=None)
        elapsed = time.time() - start
        timings.append(elapsed)
        label = int(data.y.item())
        if getattr(explanation, "perturbs", None) is not None:
            found += 1
            print(
                f"    [{i}] y={label} -> cf={explanation.label} "
                f"perturbs={explanation.perturbs} "
                f"prop={explanation.prop_perturbs:.3f}  {elapsed:.1f}s"
            )
        else:
            print(f"    [{i}] y={label} -> NO COUNTERFACTUAL   {elapsed:.1f}s")

    mean_t = sum(timings) / len(timings)
    print(f"    found {found}/{n_try} | mean {mean_t:.1f}s per graph")
    print(f"    -> projected cost for {len(test_dataset)} graphs "
          f"@1000 epochs: {mean_t * 10 * len(test_dataset) / 60:.0f} min")
    print("=" * 62)


if __name__ == "__main__":
    main()
