"""Validate the structure and basic invariants of a pilot output directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_graphs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            rows.append(row)
    return rows


def validate(output_dir: str | Path) -> dict[str, Any]:
    """Return a compact validation report or raise ``ValueError``."""
    directory = Path(output_dir)
    required = [
        directory / name
        for name in ("graphs.jsonl", "communities.csv", "edges.csv", "summary.json")
    ]
    missing = [str(path.name) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"missing required output files: {', '.join(missing)}")

    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    graphs = _load_graphs(directory / "graphs.jsonl")
    if summary.get("num_graphs") != len(graphs):
        raise ValueError(
            f"summary num_graphs={summary.get('num_graphs')} but graphs.jsonl has {len(graphs)} rows"
        )

    graph_ids = [row.get("graph_id") for row in graphs]
    if len(graph_ids) != len(set(graph_ids)):
        raise ValueError("graphs.jsonl contains duplicate graph_id values")
    for row in graphs:
        communities = row.get("communities")
        if not isinstance(communities, list) or not communities:
            raise ValueError(f"graph {row.get('graph_id')} has no community rows")
        edge_interventions = row.get("edge_interventions", [])
        if not isinstance(edge_interventions, list):
            raise ValueError(f"graph {row.get('graph_id')} has invalid edge_interventions")
        expected_calls = 1 + len(communities) + len(edge_interventions)
        if row.get("forward_calls") != expected_calls:
            raise ValueError(
                f"graph {row.get('graph_id')} reports {row.get('forward_calls')} forward calls; "
                f"expected {expected_calls}"
            )

    return {
        "valid": True,
        "num_graphs": len(graphs),
        "total_forward_calls": sum(int(row["forward_calls"]) for row in graphs),
        "files": [path.name for path in required],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GECo-CF pilot outputs")
    parser.add_argument("--input-dir", required=True, type=Path)
    args = parser.parse_args()
    report = validate(args.input_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
