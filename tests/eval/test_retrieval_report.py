from eval.retrieval_report import render_retrieval_markdown


def _summary(mode: str, groundedness: float | None) -> dict[str, object]:
    return {
        "mode": mode,
        "precision_at_k": 0.2,
        "recall_at_k": 0.9,
        "mrr": 0.8,
        "groundedness": groundedness,
        "status": "completed",
    }


def test_groundedness_note_is_omitted_when_judge_results_exist() -> None:
    markdown = render_retrieval_markdown(
        [_summary("dense", 0.96), _summary("hybrid", 0.93)],
        top_k=5,
        embedding_model="test-model",
    )

    assert "Groundedness is `n/a`" not in markdown
    assert "96.00%" in markdown
    assert "unavailable reranker" in markdown


def test_groundedness_note_is_present_when_judge_result_is_missing() -> None:
    markdown = render_retrieval_markdown(
        [_summary("dense", None), _summary("hybrid", 0.93)],
        top_k=5,
        embedding_model="test-model",
    )

    assert "Groundedness is `n/a` only where" in markdown
    assert "| n/a |" in markdown
