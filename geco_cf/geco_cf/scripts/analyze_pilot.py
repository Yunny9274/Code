"""Compute paired bootstrap effects for a completed pilot run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _graphs(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _effect(values: np.ndarray, rng: np.random.Generator, repeats: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("cannot analyze an empty result set")
    samples = values[rng.integers(0, values.size, size=(repeats, values.size))]
    means = samples.mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def analyze(input_dir: str | Path, bootstrap_repeats: int = 5000, seed: int = 0) -> dict[str, Any]:
    directory = Path(input_dir)
    rows = _graphs(directory / "graphs.jsonl")
    if not rows:
        raise ValueError("graphs.jsonl contains no rows")
    rng = np.random.default_rng(seed)
    effects = {
        "top_minus_largest_margin_drop": _effect(
            [row["top_margin_drop"] - row["largest_margin_drop"] for row in rows],
            rng,
            bootstrap_repeats,
        ),
        "top_minus_random_margin_drop": _effect(
            [row["top_margin_drop"] - row["random_mean_margin_drop"] for row in rows],
            rng,
            bootstrap_repeats,
        ),
        "top_minus_largest_flip": _effect(
            [float(row["top_flipped"]) - float(row["largest_flipped"]) for row in rows],
            rng,
            bootstrap_repeats,
        ),
        "top_minus_random_flip": _effect(
            [float(row["top_flipped"]) - row["random_flip_rate"] for row in rows],
            rng,
            bootstrap_repeats,
        ),
        "top_minus_largest_top_k_hit": _effect(
            [
                float(row["top_contains_top_k_edge"])
                - float(row["largest_contains_top_k_edge"])
                for row in rows
            ],
            rng,
            bootstrap_repeats,
        ),
        "top_minus_random_top_k_hit": _effect(
            [
                float(row["top_contains_top_k_edge"])
                - row["random_top_k_edge_hit_rate"]
                for row in rows
            ],
            rng,
            bootstrap_repeats,
        ),
    }
    return {
        "num_graphs": len(rows),
        "bootstrap_repeats": bootstrap_repeats,
        "seed": seed,
        "effects": effects,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GECo-CF pilot effects")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.input_dir, args.bootstrap_repeats, args.seed)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
