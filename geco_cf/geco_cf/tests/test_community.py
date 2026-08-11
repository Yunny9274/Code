import networkx as nx

from geco_cf.community import canonical_edge, detect_communities, partition_edges


def test_partition_internal_and_boundary_edges():
    graph = nx.Graph([(0, 1), (1, 2), (2, 3), (0, 2)])
    parts = partition_edges(graph, {0, 1})
    assert parts.internal == frozenset({(0, 1)})
    assert parts.boundary == frozenset({(0, 2), (1, 2)})
    assert canonical_edge(3, 2) == (2, 3)


def test_edgeless_graph_has_singleton_communities():
    graph = nx.empty_graph(3)
    assert detect_communities(graph) == [frozenset({0}), frozenset({1}), frozenset({2})]


def test_louvain_is_deterministic_for_fixed_seed():
    graph = nx.karate_club_graph()
    assert detect_communities(graph, "louvain", seed=7) == detect_communities(
        graph, "louvain", seed=7
    )
