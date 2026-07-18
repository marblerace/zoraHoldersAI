from __future__ import annotations

from pathlib import Path

from eval.run import load_golden_set


def test_golden_set_includes_adversarial_sql_and_clarification_cases() -> None:
    cases = load_golden_set(Path(__file__).parents[2] / "eval" / "golden_set.jsonl")
    by_id = {case.id: case for case in cases}

    assert len(cases) == 44
    assert len(by_id) == len(cases)
    assert all(by_id[case_id].reference_sql for case_id in ("q039", "q040", "q041"))
    assert all(by_id[case_id].check_type == "clarification" for case_id in ("q042", "q043", "q044"))
