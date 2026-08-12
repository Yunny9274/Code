# GECo-CF

Independent research code for community-guided coarse-to-fine counterfactual
explanations. The original `geco_explainer` and `cf-gnnexplainer-main`
directories are treated as read-only baselines.

The first milestone is deliberately small: test whether a community
intervention score identifies counterfactually sensitive regions better than
random and largest-community controls. It does **not** yet claim to implement
the complete GECo-CF optimizer.

The staged protocol, server commands, preregistered metrics, and reporting
contract are in [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md). Start with
Stage 0 locally, then report the frozen-model accuracy and a 10-graph smoke
run before launching a full server job.

For the server's first action, use the read-only preflight in
[docs/SERVER_GUIDE.md](docs/SERVER_GUIDE.md).

## Pilot definition

For a correctly classified graph with original class `y0`, define the logit
margin

```text
m(G) = z_y0(G) - max(z_c(G), c != y0)
```

For community `C`, `A(C)` is the union of its internal and boundary edges. The
pilot score is

```text
s(C) = [m(G) - m(G \ A(C))] / |A(C)|^alpha
```

The default is `alpha=0.5`. Higher scores mean that removing edges associated
with the community reduces the original prediction margin more strongly.

## Layout

```text
src/geco_cf/community.py       community detection and edge partitioning
src/geco_cf/predictors.py      framework-neutral and PyG prediction adapters
src/geco_cf/pilot.py           per-graph pilot evaluation
src/geco_cf/results.py         JSONL/CSV output
scripts/run_pilot.py           experiment entry point
scripts/train_mutagenicity.py  real-data predictor training entry point
tests/                         deterministic unit tests
configs/pilot_ba2motifs.json   first-stage defaults
configs/pilot_mutagenicity_*   Mutagenicity Stage-1 configurations
```

## Installation

Use a Python environment in which PyTorch and PyG already work:

```powershell
cd D:\Yun\code\geco_cf
python -m pip install -e .
```

Development checks additionally use pytest:

```powershell
python -m pip install -e ".[dev]"
pytest
```

## Dataset/model integration contract

The runner intentionally does not import either baseline repository. Supply a
factory as `package.module:function`. The function receives the parsed JSON
configuration and returns:

```python
dataset, predictor = build_experiment(config)
```

- `dataset` is an iterable of graph objects.
- `predictor(graph, removed_edges)` returns one-dimensional class logits.
- Each graph must expose an undirected `edge_index`, or be a NetworkX graph.
- A graph label can be supplied through `graph.y`.

For a PyG model, wrap it with `geco_cf.predictors.PyGGraphPredictor`. A minimal
adapter template is provided in `examples/ba2motifs_adapter_template.py`.

Run the pilot with:

```powershell
python scripts/run_pilot.py `
  --config configs/pilot_ba2motifs.json `
  --factory your_adapter:build_experiment `
  --output-dir results/pilot_ba2motifs
```

The output contains `graphs.jsonl`, `communities.csv`, `edges.csv`, and
`summary.json`. `edges.csv` records the single-edge scan used to test whether
high-scoring communities contain the most sensitive edges.
Only correctly classified graphs are included by default.

Validate a completed run before sharing it:

```powershell
python scripts/validate_pilot.py --input-dir results/pilot_ba2motifs
```

Compute paired bootstrap effects before comparing seeds:

```powershell
python scripts/analyze_pilot.py `
  --input-dir results/pilot_ba2motifs `
  --output results/pilot_ba2motifs/analysis.json
```

`summary.json` includes community suitability diagnostics (community count,
single-community rate, modularity, and cross-community edge ratio), the three
control outcomes, forward-call count, and end-to-end runtime. Runtime currently
includes graph conversion, community detection, scoring, and aggregation.

## Reproducibility boundary

The predictor checkpoint, dataset split, eligible graph IDs, and seed must be
identical for every method. Tune `alpha` on validation data only. Do not choose
it after inspecting test results.

## Real-data Stage-1 replication: Mutagenicity

Mutagenicity is the first real-data replication of the community-localization
hypothesis. It uses a separate three-layer GIN graph classifier and a frozen,
stratified 80/10/10 split. It remains a Stage-1 experiment: the single-edge
scan is an evaluation oracle, and no counterfactual optimizer is invoked.

On the server, train the predictor first:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_mutagenicity.py \
  --data-root data/Mutagenicity \
  --output-dir checkpoints/mutagenicity_seed0 \
  --seed 0 \
  --device cuda:0
```

Then run the 10-graph smoke configuration, validate it, and analyze it before
launching `pilot_mutagenicity_seed0.json`, `seed1.json`, and `seed2.json`.
Use `examples.mutagenicity_adapter:build_experiment` as the factory. The
checkpoint, split, and all pilot settings except the pilot seed must remain
fixed across those three runs.
