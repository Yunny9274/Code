"""Community-intervention hypothesis test used before full CF optimization."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
import networkx as nx

from .community import canonical_edges, detect_communities, partition_edges, to_networkx_undirected
from .predictors import GraphPredictor, class_margin


@dataclass(frozen=True)
class PilotConfig:
    alpha: float = 0.5
    community_method: str = "louvain"
    random_repeats: int = 30
    score_individual_edges: bool = True
    top_k_edges: int = 5
    correct_only: bool = True
    min_original_margin: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.random_repeats < 1:
            raise ValueError("random_repeats must be positive")
        if self.top_k_edges < 1:
            raise ValueError("top_k_edges must be positive")


def _average_ranks(values: list[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    order = np.argsort(values_array, kind="stable")
    ranks = np.empty(len(values_array), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values_array[order[end]] == values_array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(values_a: list[float], values_b: list[float]) -> float | None:
    if len(values_a) < 2 or len(values_a) != len(values_b):
        return None
    ranks_a = _average_ranks(values_a)
    ranks_b = _average_ranks(values_b)
    if np.ptp(ranks_a) == 0 or np.ptp(ranks_b) == 0:
        return None
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def _label(graph: Any) -> int | None:
    value = getattr(graph, "y", None)
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)[0]
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def evaluate_graph(
    graph: Any,
    predictor: GraphPredictor,
    config: PilotConfig,
    graph_id: int | str,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    started_at = perf_counter()
    original_logits = np.asarray(predictor(graph, ()), dtype=float).reshape(-1)
    original_prediction = int(original_logits.argmax())
    label = _label(graph)
    original_margin = class_margin(original_logits, original_prediction)
    if config.correct_only and label is not None and original_prediction != label:
        return None
    if original_margin < config.min_original_margin:
        return None

    nx_graph = to_networkx_undirected(graph)
    communities = detect_communities(nx_graph, config.community_method, config.seed)
    node_to_community = {
        node: community_id
        for community_id, community in enumerate(communities)
        for node in community
    }
    cross_edges = sum(
        node_to_community[int(u)] != node_to_community[int(v)] for u, v in nx_graph.edges
    )
    cross_edge_ratio = cross_edges / max(nx_graph.number_of_edges(), 1)
    modularity = (
        float(nx.community.modularity(nx_graph, communities))
        if nx_graph.number_of_edges() and communities
        else 0.0
    )
    rows: list[dict[str, Any]] = []
    affected_by_community: list[frozenset[tuple[int, int]]] = []
    for community_id, nodes in enumerate(communities):
        edge_groups = partition_edges(nx_graph, nodes)
        affected = edge_groups.affected
        affected_by_community.append(affected)
        perturbed_logits = np.asarray(predictor(graph, affected), dtype=float).reshape(-1)
        perturbed_prediction = int(perturbed_logits.argmax())
        perturbed_margin = class_margin(perturbed_logits, original_prediction)
        margin_drop = original_margin - perturbed_margin
        denominator = max(len(affected), 1) ** config.alpha
        rows.append(
            {
                "community_id": community_id,
                "nodes": sorted(nodes),
                "num_nodes": len(nodes),
                "num_internal_edges": len(edge_groups.internal),
                "num_boundary_edges": len(edge_groups.boundary),
                "num_affected_edges": len(affected),
                "perturbed_prediction": perturbed_prediction,
                "perturbed_margin": perturbed_margin,
                "margin_drop": margin_drop,
                "normalized_score": margin_drop / denominator,
                "flipped": perturbed_prediction != original_prediction,
            }
        )

    if not rows:
        return None

    edge_rows: list[dict[str, Any]] = []
    if config.score_individual_edges:
        for edge in sorted(canonical_edges(nx_graph.edges)):
            perturbed_logits = np.asarray(predictor(graph, [edge]), dtype=float).reshape(-1)
            perturbed_prediction = int(perturbed_logits.argmax())
            perturbed_margin = class_margin(perturbed_logits, original_prediction)
            edge_rows.append(
                {
                    "edge": list(edge),
                    "perturbed_prediction": perturbed_prediction,
                    "perturbed_margin": perturbed_margin,
                    "margin_drop": original_margin - perturbed_margin,
                    "flipped": perturbed_prediction != original_prediction,
                }
            )

    edge_drop = {tuple(row["edge"]): row["margin_drop"] for row in edge_rows}
    ranked_edges = sorted(edge_drop, key=lambda edge: (-edge_drop[edge], edge))
    top_k_edges = set(ranked_edges[: config.top_k_edges])
    best_edge = ranked_edges[0] if ranked_edges else None
    for row, affected in zip(rows, affected_by_community, strict=True):
        affected_drops = [edge_drop[edge] for edge in affected if edge in edge_drop]
        row["best_single_edge_margin_drop"] = max(affected_drops) if affected_drops else None
        row["contains_global_best_edge"] = best_edge in affected if best_edge is not None else False
        row["contains_top_k_edge"] = bool(affected & top_k_edges)

    top = max(rows, key=lambda row: (row["normalized_score"], -row["community_id"]))
    largest = max(rows, key=lambda row: (row["num_nodes"], -row["community_id"]))
    random_indices = rng.integers(0, len(rows), size=config.random_repeats)
    random_drops = [rows[int(index)]["margin_drop"] for index in random_indices]
    random_flips = [rows[int(index)]["flipped"] for index in random_indices]
    random_top_k_hits = [rows[int(index)]["contains_top_k_edge"] for index in random_indices]
    comparable_rows = [row for row in rows if row["best_single_edge_margin_drop"] is not None]
    community_edge_spearman = _spearman(
        [row["normalized_score"] for row in comparable_rows],
        [row["best_single_edge_margin_drop"] for row in comparable_rows],
    )
    return {
        "graph_id": graph_id,
        "label": label,
        "original_prediction": original_prediction,
        "original_margin": original_margin,
        "num_nodes": nx_graph.number_of_nodes(),
        "num_edges": nx_graph.number_of_edges(),
        "num_communities": len(rows),
        "modularity": modularity,
        "cross_edge_ratio": cross_edge_ratio,
        "top_community_id": top["community_id"],
        "top_margin_drop": top["margin_drop"],
        "top_flipped": top["flipped"],
        "largest_community_id": largest["community_id"],
        "largest_margin_drop": largest["margin_drop"],
        "largest_flipped": largest["flipped"],
        "random_mean_margin_drop": float(np.mean(random_drops)),
        "random_flip_rate": float(np.mean(random_flips)),
        "top_contains_global_best_edge": top["contains_global_best_edge"],
        "top_contains_top_k_edge": top["contains_top_k_edge"],
        "largest_contains_top_k_edge": largest["contains_top_k_edge"],
        "random_top_k_edge_hit_rate": float(np.mean(random_top_k_hits)),
        "community_edge_spearman": community_edge_spearman,
        "forward_calls": 1 + len(rows) + len(edge_rows),
        "runtime_seconds": perf_counter() - started_at,
        "communities": rows,
        "edge_interventions": edge_rows,
    }


def run_pilot(
    dataset: Iterable[Any],
    predictor: GraphPredictor,
    config: PilotConfig,
    max_graphs: int | None = None,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(config.seed)
    results: list[dict[str, Any]] = []
    for position, graph in enumerate(dataset):
        graph_id = getattr(graph, "graph_id", position)
        if hasattr(graph_id, "item"):
            graph_id = graph_id.item()
        result = evaluate_graph(graph, predictor, config, graph_id, rng)
        if result is not None:
            results.append(result)
            if max_graphs is not None and len(results) >= max_graphs:
                break
    return results


def summarize(results: list[dict[str, Any]], config: PilotConfig) -> dict[str, Any]:
    if not results:
        return {"config": asdict(config), "num_graphs": 0}
    available_correlations = [
        r["community_edge_spearman"]
        for r in results
        if r["community_edge_spearman"] is not None
    ]
    return {
        "config": asdict(config),
        "num_graphs": len(results),
        "mean_num_communities": float(np.mean([r["num_communities"] for r in results])),
        "single_community_rate": float(np.mean([r["num_communities"] == 1 for r in results])),
        "mean_modularity": float(np.mean([r["modularity"] for r in results])),
        "mean_cross_edge_ratio": float(np.mean([r["cross_edge_ratio"] for r in results])),
        "top_mean_margin_drop": float(np.mean([r["top_margin_drop"] for r in results])),
        "largest_mean_margin_drop": float(np.mean([r["largest_margin_drop"] for r in results])),
        "random_mean_margin_drop": float(np.mean([r["random_mean_margin_drop"] for r in results])),
        "top_flip_rate": float(np.mean([r["top_flipped"] for r in results])),
        "largest_flip_rate": float(np.mean([r["largest_flipped"] for r in results])),
        "random_flip_rate": float(np.mean([r["random_flip_rate"] for r in results])),
        "top_global_best_edge_hit_rate": float(
            np.mean([r["top_contains_global_best_edge"] for r in results])
        ),
        "top_top_k_edge_hit_rate": float(
            np.mean([r["top_contains_top_k_edge"] for r in results])
        ),
        "largest_top_k_edge_hit_rate": float(
            np.mean([r["largest_contains_top_k_edge"] for r in results])
        ),
        "random_top_k_edge_hit_rate": float(
            np.mean([r["random_top_k_edge_hit_rate"] for r in results])
        ),
        "mean_community_edge_spearman": (
            float(np.mean(available_correlations)) if available_correlations else None
        ),
        "num_graphs_with_defined_spearman": len(available_correlations),
        "total_forward_calls": int(sum(r["forward_calls"] for r in results)),
        "mean_runtime_seconds": float(np.mean([r["runtime_seconds"] for r in results])),
    }
