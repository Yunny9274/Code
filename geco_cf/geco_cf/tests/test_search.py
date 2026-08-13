import json

import networkx as nx
import numpy as np

from geco_cf.community import canonical_edges
from geco_cf.search import (
    SearchConfig,
    evaluate_geco_cf,
    evaluate_global_greedy,
    run_search,
    summarize_search,
)
from geco_cf.search_results import write_search_results
from scripts.validate_search import validate


class ToyGraph(nx.Graph):
    y = 0


def predictor(graph, removed_edges=()):
    removed = canonical_edges(removed_edges)
    damage = 1.6 * ((0, 1) in removed) + 1.6 * ((1, 2) in removed)
    return np.array([3.0 - damage, 0.0])


def test_global_greedy_flips_and_accounts_for_queries(tmp_path):
    graph = ToyGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
    config = SearchConfig(max_edits=3, reverse_prune=True)
    result = evaluate_global_greedy(graph, predictor, config, "toy")

    assert result is not None
    assert result["success"] is True
    assert result["removed_edges"] == [[0, 1], [1, 2]]
    assert result["edit_count"] == 2
    assert result["candidate_evaluations"] == 5
    assert result["community_scoring_calls"] == 0
    assert result["prune_calls"] == 2
    assert result["forward_calls"] == 8
    assert summarize_search([result], config)["validity"] == 1.0

    write_search_results(tmp_path, [result], summarize_search([result], config))
    report = validate(tmp_path)
    assert report["valid"] is True
    assert report["total_forward_calls"] == 8


def test_global_greedy_records_budget_failure():
    graph = ToyGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
    result = evaluate_global_greedy(
        graph, predictor, SearchConfig(max_edits=1, reverse_prune=True), "failure"
    )
    assert result is not None
    assert result["success"] is False
    assert result["edit_count"] == 1
    assert result["prune_calls"] == 0


def test_run_search_can_cap_each_class_for_balanced_smoke():
    graphs = []
    for graph_id, label in enumerate([0, 0, 0, 1, 1, 1]):
        graph = ToyGraph()
        graph.y = label
        graph.graph_id = graph_id
        graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
        graphs.append(graph)

    def always_correct(graph, removed_edges=()):
        return np.array([2.0, 0.0]) if graph.y == 0 else np.array([0.0, 2.0])

    results = run_search(
        graphs,
        always_correct,
        SearchConfig(max_edits=1),
        max_graphs=4,
        max_graphs_per_class=2,
    )
    assert [result["graph_id"] for result in results] == [0, 1, 3, 4]
    assert summarize_search(results, SearchConfig(max_edits=1))["label_counts"] == {
        "0": 2,
        "1": 2,
    }
    by_label = summarize_search(results, SearchConfig(max_edits=1))["by_label"]
    assert by_label["0"]["num_graphs"] == 2
    assert by_label["0"]["validity"] == 0.0
    assert by_label["1"]["num_graphs"] == 2


def test_geco_cf_expands_from_boundary_and_counts_coarse_queries(tmp_path):
    graph = ToyGraph()
    graph.add_edges_from(
        [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (3, 5)]
    )

    def community_predictor(graph, removed_edges=()):
        removed = canonical_edges(removed_edges)
        damage = 3.2 * ((0, 1) in removed)
        return np.array([3.0 - damage, 0.0])

    config = SearchConfig(method="geco_cf", max_edits=2, reverse_prune=True, seed=0)
    result = evaluate_geco_cf(graph, community_predictor, config, "geco-toy")
    assert result is not None
    assert result["success"] is True
    assert result["removed_edges"] == [[0, 1]]
    assert result["community_scoring_calls"] == 2
    assert result["search_steps"][0]["stage"] == "top_community"
    assert result["search_steps"][0]["stages_examined"] == 2
    assert result["num_internal_edits"] == 1
    assert result["num_boundary_edits"] == 0
    assert result["forward_calls"] == (
        1 + result["community_scoring_calls"] + result["candidate_evaluations"] + result["prune_calls"]
    )

    summary = summarize_search([result], config)
    write_search_results(tmp_path, [result], summary)
    assert validate(tmp_path)["valid"] is True


def test_bridge_aware_geco_cf_prefers_non_bridge_flip():
    graph = ToyGraph()
    graph.add_edges_from([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (3, 5)])

    def bridge_predictor(graph, removed_edges=()):
        removed = canonical_edges(removed_edges)
        if (2, 3) in removed:
            return np.array([-2.0, 0.0])
        if (0, 1) in removed:
            return np.array([-1.0, 0.0])
        return np.array([3.0, 0.0])

    plain = evaluate_geco_cf(
        graph,
        bridge_predictor,
        SearchConfig(method="geco_cf", max_edits=1, bridge_policy="none"),
        "plain",
    )
    aware = evaluate_geco_cf(
        graph,
        bridge_predictor,
        SearchConfig(method="geco_cf", max_edits=1, bridge_policy="prefer_non_bridge"),
        "aware",
    )
    assert plain is not None and aware is not None
    assert plain["success"] is True and aware["success"] is True
    assert plain["removed_edges"] == [[2, 3]]
    assert plain["num_bridge_edits"] == 1
    assert aware["removed_edges"] == [[0, 1]]
    assert aware["num_bridge_edits"] == 0
    assert aware["forward_calls"] > plain["forward_calls"]
