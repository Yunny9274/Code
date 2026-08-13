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
- `label_counts` is exactly `{"0": 5, "1": 5}`;
- `0 <= validity <= 1` and `num_successes <= num_graphs`;
- successful graphs have `edit_count <= 5`;
- `forward_calls = 1 + candidate_evaluations + prune_calls` for every graph;
- failed graphs remain in `graphs.jsonl` rather than being discarded;
- there are no NaN values or tracebacks.

The per-class cap is important because the fixed BA2Motifs test IDs are sorted
and the first eligible graphs can all belong to one class. It affects only
smoke-test selection; full evaluations retain every eligible test graph.

## Later checkpoints

After this smoke test is reviewed, the next implementation checkpoint is the
community-guided candidate expansion using the same output contract. Only
after both algorithms pass matched smoke tests will we run BA2Motifs fully,
add Random-CF and ablations, and proceed to Mutagenicity and multiple
predictor seeds.

## Validation-only budget diagnostic

The balanced five-edge smoke exposed a class-direction asymmetry: all sampled
class-1 graphs flipped with one deletion, while no sampled class-0 graph
flipped within five deletions. Do not increase the test budget in response.
Use the fixed validation split and the same five examples per class to compare
budgets 5 and 30:

```bash
for budget in 5 30; do
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python scripts/run_search.py \
    --config configs/global_greedy_ba2motifs_val_budget${budget}.json \
    --factory examples.ba2motifs_adapter:build_experiment \
    --output-dir results/global_greedy_ba2motifs_val_budget${budget}
  python scripts/validate_search.py \
    --input-dir results/global_greedy_ba2motifs_val_budget${budget}
done
```

Both summaries must report `label_counts={"0": 5, "1": 5}` and include the
new `by_label` section. Budget 30 exceeds the number of unique undirected
edges in these graphs, so it follows the greedy trajectory until a flip or
until no edge remains. If class 0 still has zero validity, increasing a
deletion budget cannot repair the observed directionality along this search;
the full paper must report per-class validity and later consider edge addition
as a separate edit setting. This diagnostic is not a full validation-set
benchmark and must not be pooled with test results.

The diagnostic produced identical success sets at budgets 5 and 30: class 0
remained at 1/5 and class 1 at 5/5, while the larger budget more than doubled
the mean query count. We therefore freeze `max_edits=5` for the first matched
deletion-only comparison.

## Step 3: matched GECo-CF smoke test

The first GECo-CF prototype ranks communities using the Stage-1 intervention
score, then searches cumulative candidate stages in this order: top-community
boundary, top-community internal edges, subsequent ranked communities, and a
full-graph fallback. A stage expands when its best candidate does not reduce
the current original-class margin. Candidate edges are rescored after every
committed edit, and successful explanations are reverse-pruned.

Community-intervention forward passes are included in the total query count.
Run GECo-CF on exactly the same balanced ten test graphs and five-edge budget
as the Global-Greedy smoke:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python scripts/run_search.py \
  --config configs/geco_cf_ba2motifs_smoke.json \
  --factory examples.ba2motifs_adapter:build_experiment \
  --output-dir results/geco_cf_ba2motifs_seed0_balanced_smoke
python scripts/validate_search.py \
  --input-dir results/geco_cf_ba2motifs_seed0_balanced_smoke
cat results/geco_cf_ba2motifs_seed0_balanced_smoke/summary.json
```

Inspect one graph per class, including its community ranking and stage trace:

```bash
python - <<'PY'
import json

path = "results/geco_cf_ba2motifs_seed0_balanced_smoke/graphs.jsonl"
rows = [json.loads(line) for line in open(path, encoding="utf-8")]
for label in (0, 1):
    row = next(row for row in rows if row["label"] == label)
    print(json.dumps(row, indent=2))
PY
```

The validator must return `valid=true`, including the accounting identity
`forward_calls = 1 + community_scoring_calls + candidate_evaluations +
prune_calls`. Compare this smoke with Global-Greedy only for debugging:

- the same graph IDs and `label_counts={"0": 5, "1": 5}` must be used;
- GECo-CF should make fewer candidate evaluations on at least some graphs;
- every selected edge must identify the candidate stage that supplied it;
- `global_fallback` is allowed and is necessary for preserving validity;
- failures and class-direction asymmetry must remain visible.

Do not run the full test split yet. First report the summary, validator output,
and the two representative graph rows so that candidate expansion, query
accounting, and matched-instance selection can be reviewed.
