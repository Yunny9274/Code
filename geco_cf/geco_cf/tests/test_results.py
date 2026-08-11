from geco_cf.results import write_results
from scripts.validate_pilot import validate


def test_validate_pilot_output(tmp_path):
    result = {
        "graph_id": 0,
        "communities": [{"community_id": 0}],
        "edge_interventions": [{"edge": [0, 1], "margin_drop": 0.5}],
        "forward_calls": 3,
    }
    write_results(tmp_path, [result], {"num_graphs": 1})
    report = validate(tmp_path)
    assert report["valid"] is True
    assert report["total_forward_calls"] == 3
