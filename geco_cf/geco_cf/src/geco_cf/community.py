"""Community detection and undirected edge bookkeeping."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import networkx as nx

Edge = tuple[int, int]


def canonical_edge(u: int, v: int) -> Edge:
    """Return a stable representation of an undirected edge."""
    u, v = int(u), int(v)
    return (u, v) if u <= v else (v, u)


def canonical_edges(edges: Iterable[tuple[int, int]]) -> set[Edge]:
    return {canonical_edge(u, v) for u, v in edges if int(u) != int(v)}


def to_networkx_undirected(graph: Any) -> nx.Graph:
    """Convert a NetworkX or PyG-like graph without importing PyG."""
    if isinstance(graph, nx.Graph):
        result = nx.Graph()
        result.add_nodes_from(graph.nodes)
        result.add_edges_from(graph.edges)
        return result

    if not hasattr(graph, "edge_index"):
        raise TypeError("graph must be a NetworkX graph or expose edge_index")

    edge_index = graph.edge_index
    if hasattr(edge_index, "detach"):
        edge_index = edge_index.detach().cpu()
    if hasattr(edge_index, "tolist"):
        edge_index = edge_index.tolist()
    if len(edge_index) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")

    num_nodes = getattr(graph, "num_nodes", None)
    if num_nodes is None and hasattr(graph, "x"):
        num_nodes = len(graph.x)
    if num_nodes is None:
        flat = edge_index[0] + edge_index[1]
        num_nodes = max(flat, default=-1) + 1

    result = nx.Graph()
    result.add_nodes_from(range(int(num_nodes)))
    result.add_edges_from(zip(edge_index[0], edge_index[1]))
    result.remove_edges_from(nx.selfloop_edges(result))
    return result


def detect_communities(
    graph: nx.Graph,
    method: str = "louvain",
    seed: int = 0,
) -> list[frozenset[int]]:
    """Detect a deterministic partition, including isolated nodes."""
    if graph.number_of_nodes() == 0:
        return []
    if graph.number_of_edges() == 0:
        communities = [frozenset([int(node)]) for node in graph.nodes]
    elif method == "louvain":
        communities = [
            frozenset(int(node) for node in community)
            for community in nx.community.louvain_communities(graph, seed=seed)
        ]
    elif method == "greedy_modularity":
        communities = [
            frozenset(int(node) for node in community)
            for community in nx.community.greedy_modularity_communities(graph)
        ]
    else:
        raise ValueError(f"unsupported community method: {method}")
    return sorted(communities, key=lambda c: (min(c), len(c)))


@dataclass(frozen=True)
class CommunityEdges:
    internal: frozenset[Edge]
    boundary: frozenset[Edge]

    @property
    def affected(self) -> frozenset[Edge]:
        return self.internal | self.boundary


def partition_edges(graph: nx.Graph, community: Iterable[int]) -> CommunityEdges:
    nodes = set(int(node) for node in community)
    internal: set[Edge] = set()
    boundary: set[Edge] = set()
    for u, v in graph.edges:
        u_in, v_in = int(u) in nodes, int(v) in nodes
        if u_in and v_in:
            internal.add(canonical_edge(u, v))
        elif u_in != v_in:
            boundary.add(canonical_edge(u, v))
    return CommunityEdges(frozenset(internal), frozenset(boundary))
