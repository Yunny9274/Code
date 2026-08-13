import json

import networkx as nx
import numpy as np

from geco_cf.community import canonical_edges
from geco_cf.search import SearchConfig, evaluate_global_greedy, summarize_search
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
