"""CLI runner for the golden question-to-SQL/answer regression set."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.cache import NullAnswerCache
from agent.service import TextToSQLAgent
from app.config import get_settings
from db.schema_context import introspect_schema
from eval.judge import GroundednessJudge
from eval.report import compute_metrics, render_markdown, write_reports
from eval.scoring import expected_answer_contains, numeric_match, result_set_match
from llm.client import LLMConfigurationError, create_llm_client
from sql_guard.executor import SQLExecutor


@dataclass(frozen=True, slots=True)
class GoldenCase:
    id: str
    question: str
    reference_sql: str | None
    expected_answer_contains: list[str]
    check_type: str
    difficulty: str


def load_golden_set(path: Path) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        case = GoldenCase(
            id=payload["id"],
            question=payload["question"],
            reference_sql=payload.get("reference_sql"),
            expected_answer_contains=payload.get("expected_answer_contains", []),
            check_type=payload["check_type"],
            difficulty=payload["difficulty"],
        )
        if case.id in seen:
            raise ValueError(f"Duplicate golden case ID {case.id} at line {line_number}")
        if case.check_type not in {"result_set_match", "numeric_match", "clarification"}:
            raise ValueError(f"Unsupported check_type {case.check_type} at line {line_number}")
        if case.check_type != "clarification" and not case.reference_sql:
            raise ValueError(f"Case {case.id} requires reference_sql")
        seen.add(case.id)
        cases.append(case)
    return cases


def evaluate_case(
    case: GoldenCase,
    *,
    agent: TextToSQLAgent,
    executor: SQLExecutor,
    judge: GroundednessJudge | None,
) -> dict[str, Any]:
    reference_rows: tuple[dict[str, Any], ...] = ()
    reference_error: str | None = None
    if case.reference_sql:
        reference = executor.run(case.reference_sql)
        reference_rows = reference.rows
        reference_error = reference.error if not reference.ok else None

    try:
        answer = agent.ask(case.question)
    except Exception as exc:
        return {
            "id": case.id,
            "question": case.question,
            "difficulty": case.difficulty,
            "check_type": case.check_type,
            "reference_sql": case.reference_sql,
            "reference_error": reference_error,
            "execution_correct": False,
            "valid_sql": False,
            "answer_contains": False,
            "grounded": None,
            "latency_ms": None,
            "cost_usd": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    if case.check_type == "clarification":
        execution_correct = answer.status == "clarification" and answer.sql is None
        valid_sql: bool | None = None
    elif reference_error:
        execution_correct = False
        valid_sql = answer.status == "succeeded" and answer.sql is not None
    else:
        scorer = numeric_match if case.check_type == "numeric_match" else result_set_match
        execution_correct = answer.status == "succeeded" and scorer(
            answer.rows,
            reference_rows,
        )
        valid_sql = answer.status == "succeeded" and answer.sql is not None

    contains_score = (
        expected_answer_contains(answer.answer, case.expected_answer_contains)
        if case.expected_answer_contains
        else None
    )
    grounded: int | None = None
    judge_error: str | None = None
    if judge is not None:
        try:
            grounded, _ = judge.score(
                question=case.question,
                answer=answer.answer,
                rows=answer.rows,
            )
        except Exception as exc:
            judge_error = f"{type(exc).__name__}: {exc}"

    return {
        "id": case.id,
        "question": case.question,
        "difficulty": case.difficulty,
        "check_type": case.check_type,
        "reference_sql": case.reference_sql,
        "generated_sql": answer.sql,
        "answer": answer.answer,
        "rows": answer.rows,
        "reference_rows": reference_rows,
        "reference_error": reference_error,
        "execution_correct": execution_correct,
        "valid_sql": valid_sql,
        "answer_contains": contains_score,
        "grounded": grounded,
        "judge_error": judge_error,
        "latency_ms": answer.latency_ms,
        "cost_usd": float(answer.cost_usd) if answer.cost_usd is not None else None,
        "token_usage": answer.token_usage.to_dict(),
        "retries": answer.retries,
        "status": answer.status,
        "error": answer.error,
    }


def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        llm = create_llm_client(settings)
    except LLMConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    cases = load_golden_set(args.golden_set)
    if args.case_ids:
        selected = set(args.case_ids)
        cases = [case for case in cases if case.id in selected]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        print("No eval cases selected.", file=sys.stderr)
        return 2

    executor = SQLExecutor(settings)
    schema = introspect_schema(settings)
    agent = TextToSQLAgent(
        settings,
        llm=llm,
        executor=executor,
        cache=NullAnswerCache(),
        schema_loader=lambda _: schema,
    )
    judge = None if args.skip_judge else GroundednessJudge(llm)
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        print(f"[{index:02d}/{len(cases):02d}] {case.id}: {case.question}", flush=True)
        result = evaluate_case(case, agent=agent, executor=executor, judge=judge)
        results.append(result)
        marker = "PASS" if result["execution_correct"] else "FAIL"
        print(f"  {marker} ({result.get('latency_ms') or 0} ms)", flush=True)

    metrics = compute_metrics(results)
    json_path, markdown_path = write_reports(
        results,
        metrics,
        output_dir=args.output_dir,
        readme_path=args.readme,
    )
    summary = render_markdown(metrics, results).split("<details>", 1)[0].rstrip()
    print(f"\n{summary}\n")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")

    accuracy = metrics["execution_accuracy"]
    if not args.no_gate and (accuracy is None or accuracy < args.threshold):
        print(
            f"Regression gate failed: execution accuracy {accuracy} < {args.threshold:.2f}",
            file=sys.stderr,
        )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the text-to-SQL golden eval set")
    root = Path(__file__).resolve().parent
    parser.add_argument("--golden-set", type=Path, default=root / "golden_set.jsonl")
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    parser.add_argument("--readme", type=Path, default=root.parent / "README.md")
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.getenv("EVAL_MIN_EXECUTION_ACCURACY", "0.75")),
    )
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case", dest="case_ids", action="append")
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
