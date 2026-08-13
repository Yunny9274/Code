"""Output helpers for Stage-2 counterfactual searches."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_search_results(
    output_dir: str | Path,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "graphs.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fields = [
        "graph_id",
        "step",
        "selected_edge",
        "candidate_count",
        "margin_before",
        "margin_after",
        "margin_drop",
        "prediction_after",
        "flipped",
        "stage",
        "stage_start",
        "stages_examined",
        "selected_was_bridge",
    ]
    with (output / "steps.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for step in result["search_steps"]:
                writer.writerow({"graph_id": result["graph_id"], **step})
