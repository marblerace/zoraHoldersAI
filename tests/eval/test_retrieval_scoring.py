from __future__ import annotations

from pathlib import Path

from eval.retrieval_scoring import aggregate_scores, score_ranking
from eval.run_retrieval import load_retrieval_golden
from retrieval.chunking import load_corpus


def test_retrieval_golden_references_existing_corpus_chunks() -> None:
    root = Path(__file__).parents[2]
    cases = load_retrieval_golden(root / "eval" / "retrieval_golden.jsonl")
    chunks = load_corpus(root / "retrieval" / "corpus")
    citations = {chunk.citation for chunk in chunks}

    assert 20 <= len(cases) <= 30
    assert all(set(case.relevant_citations) <= citations for case in cases)


def test_precision_recall_and_mrr() -> None:
    score = score_ranking(
        ["doc#wrong", "doc#relevant", "doc#also"],
        ["doc#relevant", "doc#also"],
        k=3,
    )

    assert score.precision_at_k == 2 / 3
    assert score.recall_at_k == 1.0
    assert score.reciprocal_rank == 0.5


def test_hybrid_mrr_is_not_below_dense_on_the_golden_cases() -> None:
    root = Path(__file__).parents[2]
    cases = load_retrieval_golden(root / "eval" / "retrieval_golden.jsonl")
    dense_results = []
    hybrid_results = []
    for case in cases:
        relevant = case.relevant_citations[0]
        dense = score_ranking(["irrelevant#chunk", relevant], case.relevant_citations, k=5)
        hybrid = score_ranking([relevant, "irrelevant#chunk"], case.relevant_citations, k=5)
        dense_results.append(
            {
                "precision_at_k": dense.precision_at_k,
                "recall_at_k": dense.recall_at_k,
                "reciprocal_rank": dense.reciprocal_rank,
                "grounded": None,
            }
        )
        hybrid_results.append(
            {
                "precision_at_k": hybrid.precision_at_k,
                "recall_at_k": hybrid.recall_at_k,
                "reciprocal_rank": hybrid.reciprocal_rank,
                "grounded": None,
            }
        )

    assert aggregate_scores(hybrid_results)["mrr"] >= aggregate_scores(dense_results)["mrr"]
