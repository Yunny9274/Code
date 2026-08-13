"""Discrete counterfactual search baselines for Stage 2."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import networkx as nx
import numpy as np

from .community import Edge, canonical_edges, to_networkx_undirected
from .predictors import GraphPredictor, class_margin


@dataclass(frozen=True)
class SearchConfig:
    method: str = "global_greedy"
    max_edits: int = 5
    reverse_prune: bool = True
    correct_only: bool = True
    min_original_margin: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.method != "global_greedy":
            raise ValueError(f"unsupported search method: {self.method}")
        if self.max_edits < 1:
            raise ValueError("max_edits must be positive")


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


def _component_stats(graph: nx.Graph, removed: set[Edge]) -> tuple[int, float]:
    edited = graph.copy()
    edited.remove_edges_from(removed)
    if edited.number_of_nodes() == 0:
        return 0, 0.0
    components = list(nx.connected_components(edited))
    largest_ratio = max((len(component) for component in components), default=0) / edited.number_of_nodes()
    return len(components), float(largest_ratio)


def evaluate_global_greedy(
    graph: Any,
    predictor: GraphPredictor,
    config: SearchConfig,
    graph_id: int | str,
) -> dict[str, Any] | None:
    """Run exhaustive greedy edge deletion followed by optional reverse pruning."""
    started_at = perf_counter()
    nx_graph = to_networkx_undirected(graph)
    all_edges = sorted(canonical_edges(nx_graph.edges))
    original_logits = np.asarray(predictor(graph, ()), dtype=float).reshape(-1)
    forward_calls = 1
    original_prediction = int(original_logits.argmax())
    label = _label(graph)
    original_margin = class_margin(original_logits, original_prediction)
    if config.correct_only and label is not None and original_prediction != label:
        return None
    if original_margin < config.min_original_margin:
        return None

    removed: set[Edge] = set()
    steps: list[dict[str, Any]] = []
    current_logits = original_logits
    success = False
    candidate_evaluations = 0
    edit_budget = min(config.max_edits, len(all_edges))

    for step_index in range(1, edit_budget + 1):
        best_edge: Edge | None = None
        best_logits: np.ndarray | None = None
        best_margin = float("inf")
        candidate_count = 0
        margin_before = class_margin(current_logits, original_prediction)
        for edge in all_edges:
            if edge in removed:
                continue
            trial_removed = removed | {edge}
            trial_logits = np.asarray(predictor(graph, trial_removed), dtype=float).reshape(-1)
            forward_calls += 1
            candidate_evaluations += 1
            candidate_count += 1
            trial_margin = class_margin(trial_logits, original_prediction)
            if trial_margin < best_margin:
                best_edge = edge
                best_logits = trial_logits
                best_margin = trial_margin

        if best_edge is None or best_logits is None:
            break
        removed.add(best_edge)
        current_logits = best_logits
        current_prediction = int(current_logits.argmax())
        steps.append(
            {
                "step": step_index,
                "selected_edge": list(best_edge),
                "candidate_count": candidate_count,
                "margin_before": margin_before,
                "margin_after": best_margin,
                "margin_drop": margin_before - best_margin,
                "prediction_after": current_prediction,
                "flipped": current_prediction != original_prediction,
            }
        )
        if current_prediction != original_prediction:
            success = True
            break

    pre_prune_edit_count = len(removed)
    prune_calls = 0
    pruned_edges: list[list[int]] = []
    if success and config.reverse_prune:
        for edge in reversed([tuple(step["selected_edge"]) for step in steps]):
            trial_removed = removed - {edge}
            trial_logits = np.asarray(predictor(graph, trial_removed), dtype=float).reshape(-1)
            forward_calls += 1
            prune_calls += 1
            if int(trial_logits.argmax()) != original_prediction:
                removed = trial_removed
                current_logits = trial_logits
                pruned_edges.append(list(edge))

    final_prediction = int(current_logits.argmax())
    final_margin = class_margin(current_logits, original_prediction)
    success = final_prediction != original_prediction
    original_components, original_largest_ratio = _component_stats(nx_graph, set())
    final_components, final_largest_ratio = _component_stats(nx_graph, removed)
    return {
        "graph_id": graph_id,
        "label": label,
        "method": config.method,
        "original_prediction": original_prediction,
        "final_prediction": final_prediction,
        "original_margin": original_margin,
        "final_margin": final_margin,
        "success": success,
        "num_nodes": nx_graph.number_of_nodes(),
        "num_edges": len(all_edges),
        "max_edits": config.max_edits,
        "pre_prune_edit_count": pre_prune_edit_count,
        "edit_count": len(removed),
        "normalized_edit_distance": len(removed) / max(len(all_edges), 1),
        "removed_edges": [list(edge) for edge in sorted(removed)],
        "pruned_edges": pruned_edges,
        "search_steps": steps,
        "candidate_evaluations": candidate_evaluations,
        "prune_calls": prune_calls,
        "forward_calls": forward_calls,
        "original_num_components": original_components,
        "final_num_components": final_components,
        "component_increase": final_components - original_components,
        "original_largest_component_ratio": original_largest_ratio,
        "final_largest_component_ratio": final_largest_ratio,
        "runtime_seconds": perf_counter() - started_at,
    }


def run_search(
    dataset: Iterable[Any],
    predictor: GraphPredictor,
    config: SearchConfig,
    max_graphs: int | None = None,
    max_graphs_per_class: int | None = None,
) -> list[dict[str, Any]]:
    if max_graphs is not None and max_graphs < 1:
        raise ValueError("max_graphs must be positive when provided")
    if max_graphs_per_class is not None and max_graphs_per_class < 1:
        raise ValueError("max_graphs_per_class must be positive when provided")
    results: list[dict[str, Any]] = []
    class_counts: dict[int, int] = {}
    for position, graph in enumerate(dataset):
        label = _label(graph)
        if (
            max_graphs_per_class is not None
            and label is not None
            and class_counts.get(label, 0) >= max_graphs_per_class
        ):
            continue
        graph_id = getattr(graph, "graph_id", position)
        if hasattr(graph_id, "item"):
            graph_id = graph_id.item()
        result = evaluate_global_greedy(graph, predictor, config, graph_id)
        if result is not None:
            results.append(result)
            result_label = result["label"]
            if result_label is not None:
                class_counts[result_label] = class_counts.get(result_label, 0) + 1
            if max_graphs is not None and len(results) >= max_graphs:
                break
    return results


def summarize_search(results: list[dict[str, Any]], config: SearchConfig) -> dict[str, Any]:
    summary: dict[str, Any] = {"config": asdict(config), "num_graphs": len(results)}
    if not results:
        return summary
    successes = [result for result in results if result["success"]]
    labels = sorted({result["label"] for result in results if result["label"] is not None})
    by_label: dict[str, dict[str, Any]] = {}
    for label in labels:
        label_results = [result for result in results if result["label"] == label]
        label_successes = [result for result in label_results if result["success"]]
        by_label[str(label)] = {
            "num_graphs": len(label_results),
            "num_successes": len(label_successes),
            "validity": len(label_successes) / len(label_results),
            "mean_edit_count_success": (
                float(np.mean([result["edit_count"] for result in label_successes]))
                if label_successes
                else None
            ),
            "mean_forward_calls": float(
                np.mean([result["forward_calls"] for result in label_results])
            ),
        }
    summary.update(
        {
            "label_counts": {
                str(label): sum(result["label"] == label for result in results) for label in labels
            },
            "by_label": by_label,
            "num_successes": len(successes),
            "validity": len(successes) / len(results),
            "mean_edit_count_success": (
                float(np.mean([result["edit_count"] for result in successes])) if successes else None
            ),
            "mean_normalized_edit_distance_success": (
                float(np.mean([result["normalized_edit_distance"] for result in successes]))
                if successes
                else None
            ),
            "mean_forward_calls": float(np.mean([result["forward_calls"] for result in results])),
            "total_forward_calls": int(sum(result["forward_calls"] for result in results)),
            "mean_candidate_evaluations": float(
                np.mean([result["candidate_evaluations"] for result in results])
            ),
            "mean_runtime_seconds": float(np.mean([result["runtime_seconds"] for result in results])),
            "mean_component_increase_success": (
                float(np.mean([result["component_increase"] for result in successes]))
                if successes
                else None
            ),
            "mean_largest_component_ratio_change_success": (
                float(
                    np.mean(
                        [
                            result["final_largest_component_ratio"]
                            - result["original_largest_component_ratio"]
                            for result in successes
                        ]
                    )
                )
                if successes
                else None
            ),
        }
    )
    return summary
