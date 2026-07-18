from pathlib import Path

from eval.report import compute_metrics, embed_in_readme, render_markdown


def test_metrics_exclude_clarifications_from_sql_denominators() -> None:
    results = [
        {
            "id": "q1",
            "difficulty": "easy",
            "check_type": "numeric_match",
            "execution_correct": True,
            "valid_sql": True,
            "grounded": 1,
            "answer_contains": True,
            "latency_ms": 100,
            "cost_usd": 0.01,
        },
        {
            "id": "q2",
            "difficulty": "ambiguous",
            "check_type": "clarification",
            "execution_correct": False,
            "valid_sql": None,
            "grounded": 0,
            "answer_contains": None,
            "latency_ms": 300,
            "cost_usd": 0.03,
        },
    ]

    metrics = compute_metrics(results)

    assert metrics["execution_accuracy"] == 1.0
    assert metrics["valid_sql_rate"] == 1.0
    assert metrics["clarification_accuracy"] == 0.0
    assert metrics["answer_groundedness"] == 0.5
    assert metrics["mean_latency_ms"] == 200
    assert metrics["mean_cost_usd"] == 0.02
    assert "Execution accuracy" in render_markdown(metrics, results)


def test_readme_embedding_only_replaces_marked_section(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "before\n<!-- EVAL_RESULTS_START -->\nold\n<!-- EVAL_RESULTS_END -->\nafter\n"
    )

    embed_in_readme(readme, "new table\n")

    assert readme.read_text() == (
        "before\n<!-- EVAL_RESULTS_START -->\nnew table\n<!-- EVAL_RESULTS_END -->\nafter\n"
    )
