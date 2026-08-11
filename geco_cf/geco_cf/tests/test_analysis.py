import json

from scripts.analyze_pilot import analyze


def test_analyze_pilot_bootstrap(tmp_path):
    rows = [
        {
            "top_margin_drop": 2.0,
            "largest_margin_drop": 1.0,
            "random_mean_margin_drop": 0.5,
            "top_flipped": True,
            "largest_flipped": False,
            "random_flip_rate": 0.0,
            "top_contains_top_k_edge": True,
            "largest_contains_top_k_edge": False,
            "random_top_k_edge_hit_rate": 0.0,
        },
        {
            "top_margin_drop": 3.0,
            "largest_margin_drop": 1.0,
            "random_mean_margin_drop": 1.0,
            "top_flipped": False,
            "largest_flipped": False,
            "random_flip_rate": 0.0,
            "top_contains_top_k_edge": True,
            "largest_contains_top_k_edge": True,
            "random_top_k_edge_hit_rate": 0.5,
        },
    ]
    path = tmp_path / "graphs.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    report = analyze(tmp_path, bootstrap_repeats=100, seed=0)
    assert report["num_graphs"] == 2
    assert report["effects"]["top_minus_random_margin_drop"]["mean"] == 1.75
