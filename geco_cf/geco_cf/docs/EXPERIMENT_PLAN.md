# GECo-CF experiment protocol

This protocol is the working agreement for the first implementation cycle. It
is deliberately staged: a cheap falsification pilot must pass before a full
counterfactual optimizer is added.

## Research question

Given a graph that the frozen GNN classifies correctly, does a detected
community identify a region whose edge intervention reduces the original class
margin more than the largest-community and random controls?

The pilot score for community `C` is:

```text
m(G) = z_y0(G) - max(z_c(G), c != y0)
s(C) = [m(G) - m(G without A(C))] / |A(C)|^alpha
```

`A(C)` contains the internal and boundary edges of `C`. The pilot default is
Louvain with a fixed seed and `alpha=0.5`. The model is frozen; only the graph
edges are changed.

## Hypotheses and primary metrics

- H1: the top-scoring community has a larger margin drop than random selection.
- H2: the top-scoring community has a larger margin drop than the
  largest-community control.
- H3: the top-scoring community has a higher prediction-flip rate than both
  controls.

Primary metrics are per-graph margin drop and flip rate. The pilot also scans
single-edge interventions, so report whether the top community contains the
globally best edge or any of the top five edges, plus Spearman correlation
between community score and best single-edge contribution. Report the paired
differences (top minus random, top minus largest), bootstrap 95% confidence
intervals, and the number of eligible graphs. Runtime and forward-call count
are secondary metrics.

## Fixed comparison contract

Every method must use the same:

1. dataset version, split, graph ordering, and eligible graph IDs;
2. trained checkpoint, preprocessing, class mapping, and device precision;
3. random seeds (at least 0, 1, and 2 for the final pilot);
4. original-correctness rule and margin definition.

Do not tune `alpha` or choose examples after looking at test results. Use a
validation split for any tuning, then freeze the setting before test evaluation.

## Staged execution

### Stage 0: local, CPU-only validation

Run from `geco_cf/geco_cf`:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python scripts/validate_pilot.py --help
```

This stage tests community bookkeeping, deterministic scoring, and output
validation. It does not require torch or torch-geometric.

### Stage 1: reproduce the frozen model on the server

Reproduce the BA2Motifs (or the dataset selected in the adapter) checkpoint
using the original repository's code. Record the commit/version, split seed,
checkpoint hash, test graph IDs, and test accuracy. Do not start GECo-CF
optimization until the reproduced accuracy is within the agreed tolerance of
the baseline result.

Complete `examples/ba2motifs_adapter_template.py` with that exact dataset and
model. The factory must return `(test_dataset, predictor)`, where `predictor`
returns one-dimensional **pre-softmax logits** and accepts
`predictor(graph, removed_edges)`.

### Stage 2: community intervention pilot

On the server, after installing the package and its PyG environment:

```bash
python -m pip install -e '.[dev,pyg]'
python scripts/run_pilot.py \
  --config configs/pilot_ba2motifs.json \
  --factory examples.ba2motifs_adapter_template:build_experiment \
  --output-dir results/pilot_ba2motifs_seed0
python scripts/validate_pilot.py \
  --input-dir results/pilot_ba2motifs_seed0
```

The template is intentionally incomplete until the checkpoint is reproduced.
Use the real adapter module in the command once it is filled in. Repeat with
seeds 1 and 2, using separate output directories. Send back each
`summary.json`, the validation output, and the model accuracy.

The pilot is considered promising only if the paired top-vs-control effects
are positive on the pre-registered primary metric. A negative or null result
is useful: it means we revise the community definition before implementing a
larger optimizer.

### Stage 3: full coarse-to-fine CF optimizer

Only after Stage 2, implement the optimizer in a new module. It must expose
the same predictor/checkpoint contract and produce, for each eligible graph:

- the original and counterfactual predictions;
- removed edge IDs and whether each edge is internal or boundary;
- number of forward passes, optimization steps, runtime, and success flag;
- final margin, flip status, and graph-edit size.

The first implementation is deletion-only. Compare against Random and a
Global-Greedy-CF search under the same forward-call/edit budget. For node-level
tasks also compare CF-GNNExplainer/Clean CF-GNN; for graph-level tasks add
CLEAR or another graph-CF baseline when its interface is reproducible. GECo-CF
must be evaluated with community guidance enabled and with guidance ablated.

### Stage 4: ablations and generalization

Run one-factor-at-a-time ablations on validation data, then freeze settings for
test data:

- `alpha` in `{0, 0.5, 1}`;
- internal-only versus internal-plus-boundary `A(C)`;
- greedy modularity versus Louvain (fixed seed);
- community guidance disabled (edge-level or random initialization);
- graph classification in this order: BA2Motifs, then MUTAG;
- node classification afterwards: Tree-Cycles, Tree-Grid, and BA-Shapes
  (the local `syn4`, `syn5`, and `syn1` baselines), with labels and model
  interfaces adapted consistently.

For the full method report Validity, mean edit count, sparsity, runtime,
forward calls, candidate compression, internal-versus-boundary edit ratio,
and modularity/connectedness damage. Keep graph- and node-classification
tables separate.

## Required run manifest

For every output directory, keep a small text or JSON manifest containing:

```text
dataset and version
split and eligible graph IDs
model/checkpoint path and SHA256
code commit or archive hash
python, torch, and torch-geometric versions
seed, alpha, community method, random repeats
device and dtype
```

The output files are `graphs.jsonl`, `communities.csv`, `edges.csv`, and
`summary.json`. `edges.csv` contains the single-edge scan used for the RQ1
coverage and correlation metrics.
Never merge results from different manifests into one table.

## What to report back after each step

Report the exact command, exit status, model accuracy, number of eligible
graphs, `summary.json`, and any traceback. For a long server run, first send a
10-graph smoke-run result; only then run the full test split.

## Current checkpoint

The repository is ready for Stage 0. The next action is to run the local tests,
then reproduce one frozen BA2Motifs checkpoint on the server and fill the
adapter. No claim about full GECo-CF performance should be made before the
pilot has been completed.
