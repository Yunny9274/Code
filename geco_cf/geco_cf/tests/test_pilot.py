import networkx as nx
import numpy as np
import pytest

from geco_cf.community import canonical_edges
from geco_cf.pilot import PilotConfig, evaluate_graph, summarize


class ToyGraph(nx.Graph):
    y = 0


def predictor(graph, removed_edges=()):
    removed = canonical_edges(removed_edges)
    critical = {(0, 1), (1, 2), (0, 2)}
    damage = len(removed & critical)
    return np.array([3.0 - 1.2 * damage, 0.0])


def test_top_community_finds_critical_triangle():
    graph = ToyGraph()
    graph.add_edges_from([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (3, 5)])
    config = PilotConfig(alpha=0.5, random_repeats=10, seed=0)
    result = evaluate_graph(graph, predictor, config, graph_id="toy", rng=np.random.default_rng(0))
    assert result is not None
    assert result["top_community_id"] == 0
    assert result["top_margin_drop"] > 0
    assert result["top_contains_global_best_edge"]
    assert result["top_contains_top_k_edge"]
    assert result["community_edge_spearman"] == pytest.approx(1.0)
    assert result["forward_calls"] == 10
    assert summarize([result], config)["num_graphs"] == 1
