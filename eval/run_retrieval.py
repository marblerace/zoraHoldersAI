"""Run a real dense/hybrid/reranked retrieval ablation against PostgreSQL."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_settings
from eval.judge import GroundednessJudge
from eval.retrieval_report import render_retrieval_markdown, write_retrieval_reports
from eval.retrieval_scoring import aggregate_scores, score_ranking
from llm.client import LLMConfigurationError, create_llm_client
from llm.types import LLMClient, Message
from retrieval.service import HybridRetriever, RetrievalMode


@dataclass(frozen=True, slots=True)
class RetrievalGoldenCase:
    id: str
    question: str
    relevant_citations: tuple[str, ...]


def load_retrieval_golden(path: Path) -> list[RetrievalGoldenCase]:
    cases: list[RetrievalGoldenCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        citations = tuple(payload.get("relevant_citations") or ())
        case = RetrievalGoldenCase(payload["id"], payload["question"], citations)
        if case.id in seen:
            raise ValueError(f"Duplicate retrieval case {case.id} at line {line_number}")
        if not case.question.strip() or not citations:
            raise ValueError(f"Retrieval case {case.id} requires a question and citations")
        seen.add(case.id)
        cases.append(case)
    return cases


def evaluate_mode(
    cases: list[RetrievalGoldenCase],
    *,
    retriever: HybridRetriever,
    mode: RetrievalMode,
    top_k: int,
    answer_llm: LLMClient | None = None,
    judge: GroundednessJudge | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in cases:
        retrieval = retriever.search(case.question, top_k=top_k, mode=mode)
        ranking = score_ranking(retrieval.citations, case.relevant_citations, k=top_k)
        answer = None
        grounded = None
        judge_error = None
        if answer_llm is not None and judge is not None and retrieval.hits:
            evidence = [hit.to_dict() for hit in retrieval.hits]
            try:
                answer = _answer_from_evidence(answer_llm, case.question, evidence)
                grounded, _ = judge.score(
                    question=case.question,
                    answer=answer,
                    rows=tuple(evidence),
                )
            except Exception as error:
                judge_error = f"{type(error).__name__}: {error}"
        results.append(
            {
                "id": case.id,
                "question": case.question,
                "mode": mode,
                "relevant_citations": case.relevant_citations,
                "retrieved_citations": retrieval.citations,
                "precision_at_k": ranking.precision_at_k,
                "recall_at_k": ranking.recall_at_k,
                "reciprocal_rank": ranking.reciprocal_rank,
                "latency_ms": retrieval.latency_ms,
                "error": retrieval.error,
                "warnings": retrieval.warnings,
                "reranked": retrieval.reranked,
                "answer": answer,
                "grounded": grounded,
                "judge_error": judge_error,
            }
        )
    summary = {"mode": mode, **aggregate_scores(results)}
    errors = sum(bool(result["error"]) for result in results)
    warnings = sum(bool(result["warnings"]) for result in results)
    if mode == "hybrid_rerank" and not any(result["reranked"] for result in results):
        summary.update(
            {
                "precision_at_k": None,
                "recall_at_k": None,
                "mrr": None,
                "groundedness": None,
                "status": "unavailable (reranker not loaded)",
            }
        )
    elif errors == len(results):
        summary["status"] = "failed"
    elif errors or warnings:
        summary["status"] = f"completed with {errors} errors / {warnings} warnings"
    else:
        summary["status"] = "completed"
    return results, summary


def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    updates: dict[str, Any] = {}
    if args.allow_model_download:
        updates["fastembed_local_files_only"] = False
    updates["reranker_provider"] = "none" if args.skip_reranker else "fastembed"
    settings = settings.model_copy(update=updates)

    cases = load_retrieval_golden(args.golden_set)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("No retrieval cases selected.", file=sys.stderr)
        return 2

    answer_llm = None
    judge = None
    if args.judge:
        try:
            answer_llm = create_llm_client(settings)
        except LLMConfigurationError as error:
            print(f"Judge configuration error: {error}", file=sys.stderr)
            return 2
        judge = GroundednessJudge(answer_llm)

    retriever = HybridRetriever(settings)
    all_results: dict[str, list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    for mode in ("dense", "hybrid", "hybrid_rerank"):
        print(f"Running {mode} over {len(cases)} cases...", flush=True)
        results, summary = evaluate_mode(
            cases,
            retriever=retriever,
            mode=mode,
            top_k=args.top_k,
            answer_llm=answer_llm,
            judge=judge,
        )
        all_results[mode] = results
        summaries.append(summary)

    embedding_model = retriever.embedding_model
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "golden_set": str(args.golden_set),
        "top_k": args.top_k,
        "embedding_model": embedding_model,
        "summaries": summaries,
        "results": all_results,
    }
    json_path, markdown_path = write_retrieval_reports(
        payload,
        output_dir=args.output_dir,
        readme_path=args.readme,
    )
    markdown = render_retrieval_markdown(
        summaries,
        top_k=args.top_k,
        embedding_model=embedding_model,
    )
    print("\n" + markdown)
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")

    dense = next(item for item in summaries if item["mode"] == "dense")
    hybrid = next(item for item in summaries if item["mode"] == "hybrid")
    dense_mrr = dense.get("mrr")
    hybrid_mrr = hybrid.get("mrr")
    if not args.no_gate and (dense_mrr is None or hybrid_mrr is None or hybrid_mrr < dense_mrr):
        print("Retrieval gate failed: hybrid MRR is below dense-only MRR", file=sys.stderr)
        return 1
    return 0


def _answer_from_evidence(
    llm: LLMClient,
    question: str,
    evidence: list[dict[str, Any]],
) -> str:
    completion = llm.complete(
        [
            Message(
                role="system",
                content=(
                    "Answer only from the supplied evidence. Cite factual claims inline "
                    "as [doc_id#chunk_id]. If evidence is insufficient, say so."
                ),
            ),
            Message(
                role="user",
                content=json.dumps({"question": question, "evidence": evidence}),
            ),
        ],
        [],
    )
    return completion.text


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run the hybrid retrieval ablation")
    parser.add_argument("--golden-set", type=Path, default=root / "retrieval_golden.jsonl")
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    parser.add_argument("--readme", type=Path, default=root.parent / "README.md")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--skip-reranker", action="store_true")
    parser.add_argument("--no-gate", action="store_true")
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
