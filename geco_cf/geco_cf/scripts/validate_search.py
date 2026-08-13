"""Validate Stage-2 search outputs and accounting invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            rows.append(row)
    return rows


def validate(output_dir: str | Path) -> dict[str, Any]:
    directory = Path(output_dir)
    required = [directory / name for name in ("graphs.jsonl", "steps.csv", "summary.json")]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"missing required output files: {', '.join(missing)}")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    graphs = _load_jsonl(directory / "graphs.jsonl")
    if summary.get("num_graphs") != len(graphs):
        raise ValueError("summary num_graphs does not match graphs.jsonl")
    graph_ids = [row.get("graph_id") for row in graphs]
    if len(graph_ids) != len(set(graph_ids)):
        raise ValueError("graphs.jsonl contains duplicate graph IDs")

    for row in graphs:
        required_fields = {
            "success",
            "edit_count",
            "removed_edges",
            "search_steps",
            "community_scoring_calls",
            "candidate_evaluations",
            "prune_calls",
            "forward_calls",
        }
        missing_fields = sorted(required_fields - set(row))
        if missing_fields:
            raise ValueError(f"graph {row.get('graph_id')} is missing {missing_fields}")
        if row["edit_count"] != len(row["removed_edges"]):
            raise ValueError(f"graph {row.get('graph_id')} has inconsistent edit_count")
        expected_calls = (
            1
            + int(row["community_scoring_calls"])
            + int(row["candidate_evaluations"])
            + int(row["prune_calls"])
        )
        if row["forward_calls"] != expected_calls:
            raise ValueError(
                f"graph {row.get('graph_id')} reports {row['forward_calls']} calls; "
                f"expected {expected_calls}"
            )
        if row["success"] != (row["final_prediction"] != row["original_prediction"]):
            raise ValueError(f"graph {row.get('graph_id')} has inconsistent success status")

    return {
        "valid": True,
        "num_graphs": len(graphs),
        "num_successes": sum(bool(row["success"]) for row in graphs),
        "total_forward_calls": sum(int(row["forward_calls"]) for row in graphs),
        "files": [path.name for path in required],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GECo-CF Stage-2 search outputs")
    parser.add_argument("--input-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.input_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
