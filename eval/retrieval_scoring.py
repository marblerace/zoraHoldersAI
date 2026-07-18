"""Deterministic ranking metrics for the retrieval golden set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RankingScore:
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float


def score_ranking(
    retrieved: list[str] | tuple[str, ...],
    relevant: list[str] | tuple[str, ...],
    *,
    k: int,
) -> RankingScore:
    """Compute precision@k, recall@k, and reciprocal rank for one question."""

    if k < 1:
        raise ValueError("k must be positive")
    relevant_set = set(relevant)
    top = list(retrieved[:k])
    matched = sum(citation in relevant_set for citation in top)
    first_rank = next(
        (rank for rank, citation in enumerate(retrieved, 1) if citation in relevant_set),
        None,
    )
    return RankingScore(
        precision_at_k=matched / k,
        recall_at_k=matched / len(relevant_set) if relevant_set else 1.0,
        reciprocal_rank=1.0 / first_rank if first_rank is not None else 0.0,
    )


def aggregate_scores(results: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Average per-question metrics without fabricating values for empty runs."""

    if not results:
        return {
            "cases": 0,
            "precision_at_k": None,
            "recall_at_k": None,
            "mrr": None,
            "groundedness": None,
        }
    grounded = [item["grounded"] for item in results if item.get("grounded") is not None]
    return {
        "cases": len(results),
        "precision_at_k": sum(item["precision_at_k"] for item in results) / len(results),
        "recall_at_k": sum(item["recall_at_k"] for item in results) / len(results),
        "mrr": sum(item["reciprocal_rank"] for item in results) / len(results),
        "groundedness": sum(grounded) / len(grounded) if grounded else None,
    }
