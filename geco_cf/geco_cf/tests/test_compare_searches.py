import json

import pytest

from scripts.compare_searches import compare


def _write(directory, rows):
    directory.mkdir()
    (directory / "graphs.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )


def _row(graph_id, method, label, success, calls, candidates):
    return {
        "graph_id": graph_id,
        "method": method,
        "label": label,
        "original_prediction": label,
        "max_edits": 5,
        "num_edges": 10,
        "success": success,
        "forward_calls": calls,
        "candidate_evaluations": candidates,
        "community_scoring_calls": 2 if method == "geco_cf" else 0,
        "edit_count": 1 if success else 5,
        "component_increase": 1 if success else 4,
        "original_largest_component_ratio": 1.0,
        "final_largest_component_ratio": 0.8 if success else 0.5,
    }


def test_compare_matched_searches(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write(
        baseline,
        [_row(1, "global_greedy", 0, False, 20, 19), _row(2, "global_greedy", 1, True, 20, 18)],
    )
    _write(
        candidate,
        [_row(1, "geco_cf", 0, False, 16, 13), _row(2, "geco_cf", 1, True, 10, 6)],
    )
    report = compare(baseline, candidate)
    assert report["matched"] is True
    assert report["overall"]["validity_difference"] == 0.0
    assert report["overall"]["forward_call_reduction_rate"] == pytest.approx(0.35)
    assert report["by_label"]["1"]["both_success"] == 1


def test_compare_rejects_unmatched_graph_ids(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write(baseline, [_row(1, "global_greedy", 0, False, 20, 19)])
    _write(candidate, [_row(2, "geco_cf", 0, False, 16, 13)])
    with pytest.raises(ValueError, match="graph IDs do not match"):
        compare(baseline, candidate)
