"""Write retrieval ablation artifacts and update dedicated README markers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

README_START = "<!-- RETRIEVAL_RESULTS_START -->"
README_END = "<!-- RETRIEVAL_RESULTS_END -->"


def render_retrieval_markdown(
    summaries: list[dict[str, Any]],
    *,
    top_k: int,
    embedding_model: str,
) -> str:
    lines = [
        f"Real retrieval run using `{embedding_model}`; ranking cutoff k={top_k}.",
        "",
        f"| Configuration | Precision@{top_k} | Recall@{top_k} | MRR | Groundedness | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    labels = {
        "dense": "Dense only",
        "hybrid": "Dense + FTS (RRF)",
        "hybrid_rerank": "Hybrid + reranker",
    }
    for summary in summaries:
        lines.append(
            "| {label} | {precision} | {recall} | {mrr} | {grounded} | {status} |".format(
                label=labels.get(summary["mode"], summary["mode"]),
                precision=_rate(summary.get("precision_at_k")),
                recall=_rate(summary.get("recall_at_k")),
                mrr=_number(summary.get("mrr")),
                grounded=_rate(summary.get("groundedness")),
                status=summary.get("status", "unknown"),
            )
        )
    lines.extend(
        [
            "",
            "Groundedness is `n/a` unless the optional provider-backed judge was run. "
            "An unavailable reranker is reported explicitly and is never relabeled "
            "as a reranked run.",
            "",
        ]
    )
    return "\n".join(lines)


def write_retrieval_reports(
    payload: dict[str, Any],
    *,
    output_dir: Path,
    readme_path: Path | None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"retrieval-{timestamp}.json"
    markdown_path = output_dir / "retrieval-latest.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    markdown = render_retrieval_markdown(
        payload["summaries"],
        top_k=payload["top_k"],
        embedding_model=payload["embedding_model"],
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    if readme_path and readme_path.exists():
        embed_retrieval_in_readme(readme_path, markdown)
    return json_path, markdown_path


def embed_retrieval_in_readme(readme_path: Path, markdown: str) -> None:
    content = readme_path.read_text(encoding="utf-8")
    if README_START not in content or README_END not in content:
        return
    before, remainder = content.split(README_START, 1)
    _, after = remainder.split(README_END, 1)
    updated = f"{before}{README_START}\n{markdown.rstrip()}\n{README_END}{after}"
    readme_path.write_text(updated, encoding="utf-8")


def _rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"
