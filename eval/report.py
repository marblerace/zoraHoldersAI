"""Aggregate eval metrics and write JSON/Markdown regression artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.scoring import percentile

README_START = "<!-- EVAL_RESULTS_START -->"
README_END = "<!-- EVAL_RESULTS_END -->"


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    executable = [result for result in results if result["check_type"] != "clarification"]
    clarifications = [result for result in results if result["check_type"] == "clarification"]
    judged = [result["grounded"] for result in results if result.get("grounded") is not None]
    latencies = [
        float(result["latency_ms"]) for result in results if result.get("latency_ms") is not None
    ]
    costs = [float(result["cost_usd"]) for result in results if result.get("cost_usd") is not None]

    return {
        "cases": len(results),
        "executable_cases": len(executable),
        "execution_accuracy": _mean_bool(result.get("execution_correct") for result in executable),
        "valid_sql_rate": _mean_bool(result.get("valid_sql") for result in executable),
        "clarification_accuracy": _mean_bool(
            result.get("execution_correct") for result in clarifications
        ),
        "answer_groundedness": _mean_bool(judged),
        "answer_contains_rate": _mean_bool(result.get("answer_contains") for result in results),
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 0.95),
        "mean_cost_usd": sum(costs) / len(costs) if costs else None,
        "total_cost_usd": sum(costs) if costs else None,
    }


def write_reports(
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    output_dir: Path,
    readme_path: Path | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"{timestamp}.json"
    markdown_path = output_dir / "latest.md"
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    markdown = render_markdown(metrics, results)
    markdown_path.write_text(markdown, encoding="utf-8")
    if readme_path and readme_path.exists():
        embed_in_readme(readme_path, markdown)
    return json_path, markdown_path


def render_markdown(metrics: dict[str, Any], results: list[dict[str, Any]]) -> str:
    rows = [
        ("Execution accuracy", _format_rate(metrics["execution_accuracy"])),
        ("Valid-SQL rate", _format_rate(metrics["valid_sql_rate"])),
        ("Clarification accuracy", _format_rate(metrics["clarification_accuracy"])),
        ("Answer groundedness", _format_rate(metrics["answer_groundedness"])),
        ("Mean latency", _format_duration(metrics["mean_latency_ms"])),
        ("p95 latency", _format_duration(metrics["p95_latency_ms"])),
        ("Mean cost / query", _format_cost(metrics["mean_cost_usd"])),
    ]
    lines = [
        "| Metric | Value |",
        "|---|---:|",
        *(f"| {name} | {value} |" for name, value in rows),
        "",
        f"Cases: {metrics['cases']} total / {metrics['executable_cases']} executable.",
        "",
        "<details>",
        "<summary>Per-case results</summary>",
        "",
        "| ID | Difficulty | Check | Correct | Valid SQL | Latency | Cost |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for result in results:
        row_template = (
            "| {id} | {difficulty} | {check_type} | {correct} | {valid} | {latency} | {cost} |"
        )
        lines.append(
            row_template.format(
                id=result["id"],
                difficulty=result["difficulty"],
                check_type=result["check_type"],
                correct="✓" if result.get("execution_correct") else "✗",
                valid=(
                    "n/a"
                    if result.get("valid_sql") is None
                    else ("✓" if result["valid_sql"] else "✗")
                ),
                latency=_format_duration(result.get("latency_ms")),
                cost=_format_cost(result.get("cost_usd")),
            )
        )
    lines.extend(["", "</details>", ""])
    return "\n".join(lines)


def embed_in_readme(readme_path: Path, markdown: str) -> None:
    content = readme_path.read_text(encoding="utf-8")
    if README_START not in content or README_END not in content:
        return
    before, remainder = content.split(README_START, 1)
    _, after = remainder.split(README_END, 1)
    updated = f"{before}{README_START}\n{markdown.rstrip()}\n{README_END}{after}"
    readme_path.write_text(updated, encoding="utf-8")


def _mean_bool(values: Any) -> float | None:
    resolved = [bool(value) for value in values if value is not None]
    return sum(resolved) / len(resolved) if resolved else None


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _format_duration(value: float | None) -> str:
    return "n/a" if value is None else f"{value / 1000:.2f} s"


def _format_cost(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.6f}"
