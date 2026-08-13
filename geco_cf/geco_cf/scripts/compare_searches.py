"""Compare two Stage-2 searches on exactly matched graph instances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[int | str, dict[str, Any]]:
    rows: dict[int | str, dict[str, Any]] = {}
    with (path / "graphs.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            graph_id = row.get("graph_id")
            if graph_id in rows:
                raise ValueError(f"duplicate graph_id {graph_id!r} in {path}")
            rows[graph_id] = row
    return rows


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _group_report(
    baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if not baseline_rows:
        return {"num_graphs": 0}
    baseline_success = [bool(row["success"]) for row in baseline_rows]
    candidate_success = [bool(row["success"]) for row in candidate_rows]
    joint_success_indices = [
        index
        for index, (base, candidate) in enumerate(zip(baseline_success, candidate_success, strict=True))
        if base and candidate
    ]
    base_forward = _mean(baseline_rows, "forward_calls")
    candidate_forward = _mean(candidate_rows, "forward_calls")
    base_candidates = _mean(baseline_rows, "candidate_evaluations")
    candidate_candidates = _mean(candidate_rows, "candidate_evaluations")
    report: dict[str, Any] = {
        "num_graphs": len(baseline_rows),
        "baseline_validity": float(np.mean(baseline_success)),
        "candidate_validity": float(np.mean(candidate_success)),
        "validity_difference": float(np.mean(candidate_success) - np.mean(baseline_success)),
        "both_success": sum(base and candidate for base, candidate in zip(baseline_success, candidate_success, strict=True)),
        "baseline_only_success": sum(base and not candidate for base, candidate in zip(baseline_success, candidate_success, strict=True)),
        "candidate_only_success": sum(candidate and not base for base, candidate in zip(baseline_success, candidate_success, strict=True)),
        "both_failed": sum(not base and not candidate for base, candidate in zip(baseline_success, candidate_success, strict=True)),
        "baseline_mean_forward_calls": base_forward,
        "candidate_mean_forward_calls": candidate_forward,
        "mean_forward_call_difference": candidate_forward - base_forward,
        "forward_call_reduction_rate": (base_forward - candidate_forward) / base_forward,
        "baseline_mean_candidate_evaluations": base_candidates,
        "candidate_mean_candidate_evaluations": candidate_candidates,
        "candidate_evaluation_reduction_rate": (
            (base_candidates - candidate_candidates) / base_candidates
        ),
        "candidate_mean_community_scoring_calls": float(
            np.mean([row.get("community_scoring_calls", 0) for row in candidate_rows])
        ),
        "num_joint_successes": len(joint_success_indices),
    }
    if joint_success_indices:
        base_joint = [baseline_rows[index] for index in joint_success_indices]
        candidate_joint = [candidate_rows[index] for index in joint_success_indices]
        report.update(
            {
                "baseline_mean_edit_count_joint_success": _mean(base_joint, "edit_count"),
                "candidate_mean_edit_count_joint_success": _mean(candidate_joint, "edit_count"),
                "mean_edit_count_difference_joint_success": (
                    _mean(candidate_joint, "edit_count") - _mean(base_joint, "edit_count")
                ),
                "baseline_mean_component_increase_joint_success": _mean(
                    base_joint, "component_increase"
                ),
                "candidate_mean_component_increase_joint_success": _mean(
                    candidate_joint, "component_increase"
                ),
                "mean_component_increase_difference_joint_success": (
                    _mean(candidate_joint, "component_increase")
                    - _mean(base_joint, "component_increase")
                ),
                "baseline_mean_largest_component_ratio_change_joint_success": float(
                    np.mean(
                        [
                            row["final_largest_component_ratio"]
                            - row["original_largest_component_ratio"]
                            for row in base_joint
                        ]
                    )
                ),
                "candidate_mean_largest_component_ratio_change_joint_success": float(
                    np.mean(
                        [
                            row["final_largest_component_ratio"]
                            - row["original_largest_component_ratio"]
                            for row in candidate_joint
                        ]
                    )
                ),
            }
        )
    return report


def compare(baseline_dir: str | Path, candidate_dir: str | Path) -> dict[str, Any]:
    baseline = _load(Path(baseline_dir))
    candidate = _load(Path(candidate_dir))
    if set(baseline) != set(candidate):
        missing_candidate = sorted(set(baseline) - set(candidate), key=str)
        missing_baseline = sorted(set(candidate) - set(baseline), key=str)
        raise ValueError(
            f"graph IDs do not match; missing candidate={missing_candidate}, "
            f"missing baseline={missing_baseline}"
        )
    graph_ids = sorted(baseline, key=str)
    for graph_id in graph_ids:
        base = baseline[graph_id]
        other = candidate[graph_id]
        for field in ("label", "original_prediction", "max_edits", "num_edges"):
            if base.get(field) != other.get(field):
                raise ValueError(
                    f"graph {graph_id!r} has mismatched {field}: "
                    f"{base.get(field)!r} != {other.get(field)!r}"
                )

    baseline_rows = [baseline[graph_id] for graph_id in graph_ids]
    candidate_rows = [candidate[graph_id] for graph_id in graph_ids]
    labels = sorted({row["label"] for row in baseline_rows if row.get("label") is not None})
    return {
        "matched": True,
        "num_graphs": len(graph_ids),
        "baseline_method": baseline_rows[0]["method"] if baseline_rows else None,
        "candidate_method": candidate_rows[0]["method"] if candidate_rows else None,
        "overall": _group_report(baseline_rows, candidate_rows),
        "by_label": {
            str(label): _group_report(
                [row for row in baseline_rows if row["label"] == label],
                [row for row in candidate_rows if row["label"] == label],
            )
            for label in labels
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare matched Stage-2 search outputs")
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.baseline_dir, args.candidate_dir)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
