from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.run import (
    GoldenCase,
    ReferenceQueryError,
    _execution_gate_failed,
    _is_safe_clarification,
    evaluate_case,
    load_golden_set,
)


def test_golden_set_includes_adversarial_sql_and_clarification_cases() -> None:
    cases = load_golden_set(Path(__file__).parents[2] / "eval" / "golden_set.jsonl")
    by_id = {case.id: case for case in cases}

    assert len(cases) == 44
    assert len(by_id) == len(cases)
    assert all(by_id[case_id].reference_sql for case_id in ("q039", "q040", "q041"))
    assert all(by_id[case_id].check_type == "clarification" for case_id in ("q042", "q043", "q044"))
    assert " OR " not in by_id["q039"].reference_sql
    assert "stored transfer amount units" in by_id["q039"].question
    assert "breaking ties" in by_id["q039"].question
    assert "breaking balance ties" in by_id["q041"].question


def test_reference_query_failure_invalidates_evaluation() -> None:
    case = GoldenCase(
        id="q-test",
        question="question",
        reference_sql="SELECT 1",
        expected_answer_contains=[],
        check_type="numeric_match",
        difficulty="easy",
    )
    executor = SimpleNamespace(
        run=lambda _: SimpleNamespace(ok=False, error="statement timeout", rows=())
    )

    with pytest.raises(ReferenceQueryError, match="q-test: statement timeout"):
        evaluate_case(case, agent=None, executor=executor, judge=None)


def test_clarification_accepts_only_safe_non_sql_behaviors() -> None:
    direct = SimpleNamespace(sql=None, rows=(), status="clarification", citations=())
    cited_rejection = SimpleNamespace(
        sql=None,
        rows=(),
        status="succeeded",
        citations=("methodology#scope",),
    )
    unsupported_answer = SimpleNamespace(sql=None, rows=(), status="succeeded", citations=())
    queried_answer = SimpleNamespace(
        sql="SELECT COUNT(*) FROM holders",
        rows=({"count": 1},),
        status="succeeded",
        citations=(),
    )

    assert _is_safe_clarification(direct) is True
    assert _is_safe_clarification(cited_rejection) is True
    assert _is_safe_clarification(unsupported_answer) is False
    assert _is_safe_clarification(queried_answer) is False


def test_sql_gate_skips_clarification_only_subsets() -> None:
    clarification_only = {"executable_cases": 0, "execution_accuracy": None}
    passing_sql = {"executable_cases": 2, "execution_accuracy": 0.9}
    failing_sql = {"executable_cases": 2, "execution_accuracy": 0.5}

    assert _execution_gate_failed(clarification_only, 0.75) is False
    assert _execution_gate_failed(passing_sql, 0.75) is False
    assert _execution_gate_failed(failing_sql, 0.75) is True
