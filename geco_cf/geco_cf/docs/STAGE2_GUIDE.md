# GECo-CF Stage 2 guide

Stage 2 evaluates whether community guidance can accelerate a discrete
counterfactual search without materially reducing validity or increasing the
number of edits. Do not run the full dataset yet. The first checkpoint is a
10-eligible-graph Global-Greedy-CF smoke test on BA2Motifs.

## Baseline definition

Global-Greedy-CF starts from the original graph and, at every step, evaluates
deleting each remaining edge in addition to the edges already removed. It
permanently deletes the edge that minimizes the original-class margin. The
search stops when the prediction flips or the edit budget is exhausted. After
a flip, reverse pruning attempts to restore deleted edges while preserving the
new prediction.

Because it scans the whole remaining edge set after every edit, this method is
an expensive reference. GECo-CF should later approach its validity and edit
count with fewer candidate evaluations and forward calls.

## Step 1: synchronize and run local checks

Use the `stage2-adaptive-search` branch or synchronize the files from commit
`stage2-adaptive-search` after they have been committed. From the server
project root:

```bash
export PYTHONPATH=src:.
python -m pytest
python scripts/run_search.py --help
python scripts/validate_search.py --help
```

The test suite must pass before using the GPU.

## Step 2: run only the BA2Motifs smoke test

Reuse the frozen BA2Motifs seed-0 checkpoint and split from Stage 1:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python scripts/run_search.py \
  --config configs/global_greedy_ba2motifs_smoke.json \
  --factory examples.ba2motifs_adapter:build_experiment \
  --output-dir results/global_greedy_ba2motifs_seed0_smoke
```

Validate the accounting invariants:

```bash
python scripts/validate_search.py \
  --input-dir results/global_greedy_ba2motifs_seed0_smoke
```

Then inspect:

```bash
cat results/global_greedy_ba2motifs_seed0_smoke/summary.json
head -n 2 results/global_greedy_ba2motifs_seed0_smoke/graphs.jsonl
```

Report all three command outputs. Do not launch the full test split until the
following checks pass:

- the validator returns `"valid": true`;
- `num_graphs` is 10;
- `0 <= validity <= 1` and `num_successes <= num_graphs`;
- successful graphs have `edit_count <= 5`;
- `forward_calls = 1 + candidate_evaluations + prune_calls` for every graph;
- failed graphs remain in `graphs.jsonl` rather than being discarded;
- there are no NaN values or tracebacks.

## Later checkpoints

After this smoke test is reviewed, the next implementation checkpoint is the
community-guided candidate expansion using the same output contract. Only
after both algorithms pass matched smoke tests will we run BA2Motifs fully,
add Random-CF and ablations, and proceed to Mutagenicity and multiple
predictor seeds.
