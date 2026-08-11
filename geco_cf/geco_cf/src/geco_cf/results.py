"""Stable, inspectable pilot outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_results(output_dir: str | Path, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "graphs.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    fields = [
        "graph_id", "community_id", "nodes", "num_nodes", "num_internal_edges",
        "num_boundary_edges", "num_affected_edges", "perturbed_prediction",
        "perturbed_margin", "margin_drop", "normalized_score", "flipped",
        "best_single_edge_margin_drop", "contains_global_best_edge", "contains_top_k_edge",
    ]
    with (output / "communities.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for community in result["communities"]:
                writer.writerow({"graph_id": result["graph_id"], **community})

    edge_fields = [
        "graph_id", "edge", "perturbed_prediction", "perturbed_margin", "margin_drop", "flipped",
    ]
    with (output / "edges.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=edge_fields)
        writer.writeheader()
        for result in results:
            for edge in result.get("edge_interventions", []):
                writer.writerow({"graph_id": result["graph_id"], **edge})
