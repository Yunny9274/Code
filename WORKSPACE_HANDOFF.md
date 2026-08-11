# GECo-CF Workspace Handoff

## Repository

- Local repository root: `code/`
- Current branch: `main`
- Initial commit: `a9ab1e0` (`Initialize GECo-CF research workspace`)
- Remote: not configured yet
- Git author currently configured as `王赟240108 <2957451296@xiaomi.com>`

The repository tracks the `geco_cf/` experiment project and the
`Springer_Lecture_Notes_in_Computer_Science/` manuscript source. It excludes
datasets, checkpoints, pilot outputs, Python caches, LaTeX build products,
literature PDFs, and the unrelated projects in the shared `code/` directory.

## Stage 1 Snapshot

The BA2Motifs pilot used a frozen seed-0 raw-adjacency GCN with max-plus-sum
readout (validation accuracy 1.00, test accuracy 0.98). Three pilot seeds (0,
1, 2) evaluated Louvain community intervention ranking against largest and
random community controls over 98 correctly classified test graphs. Top-5
sensitive-edge coverage was 0.786--0.816 for the ranked community and
0.521--0.532 for random selection. The paired bootstrap results and caveats
are written in `Springer_Lecture_Notes_in_Computer_Science/samplepaper.tex`.

This is evidence for the Stage-1 community-localization hypothesis only. The
adaptive fine-grained counterfactual search, predictor-seed robustness,
additional datasets, and full baselines remain Stage 2 work.

## After Connecting GitHub

From this directory, configure the remote and push `main`:

```powershell
git remote add origin https://github.com/<account>/<repository>.git
git push -u origin main
```

Do not send a password or access token in chat. Use Git Credential Manager,
SSH, or a GitHub login flow on the local machine.

## Planned Branching Point

After the introduction and figures are finalized and the Stage-1 snapshot is
tagged, create the second-stage branch:

```powershell
git switch -c stage2-adaptive-search
```

The branch should begin from the reviewed Stage-1 commit/tag and keep Stage-1
code and results reproducible.
