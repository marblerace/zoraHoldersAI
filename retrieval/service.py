"""Dense + PostgreSQL FTS retrieval fused with Reciprocal Rank Fusion."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import replace
from typing import Literal, Protocol

from app.config import Settings, get_settings
from observability.tracing import start_span
from retrieval.embeddings import (
    EmbeddingBackend,
    FastEmbedReranker,
    Reranker,
    create_embedding_backend,
)
from retrieval.models import RetrievalResult, SearchHit
from sql_guard.executor import ExecutionResult, SQLExecutor

RetrievalMode = Literal["dense", "sparse", "hybrid", "hybrid_rerank"]


class Executor(Protocol):
    def run(self, query: str) -> ExecutionResult: ...


class HybridRetriever:
    """Query one guarded PostgreSQL store using dense and sparse rankers."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: EmbeddingBackend | None = None,
        executor: Executor | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embedder = embedder or create_embedding_backend(self._settings)
        self._executor = executor or SQLExecutor(self._settings)
        if reranker is not None:
            self._reranker = reranker
        elif self._settings.reranker_provider == "fastembed":
            try:
                self._reranker = FastEmbedReranker(self._settings)
            except Exception:
                self._reranker = None
        else:
            self._reranker = None

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        mode: RetrievalMode = "hybrid",
    ) -> RetrievalResult:
        started = time.perf_counter()
        normalized = " ".join(query.split())
        limit = self._settings.retrieval_top_k if top_k is None else top_k
        if mode not in {"dense", "sparse", "hybrid", "hybrid_rerank"}:
            return RetrievalResult(
                query=query,
                mode=mode,
                error=f"Unsupported retrieval mode: {mode}",
                latency_ms=_elapsed_ms(started),
            )
        if not normalized:
            return RetrievalResult(
                query=query,
                mode=mode,
                error="search_docs requires a non-empty query",
                latency_ms=_elapsed_ms(started),
            )
        if limit < 1 or limit > 20:
            return RetrievalResult(
                query=query,
                mode=mode,
                error="top_k must be between 1 and 20",
                latency_ms=_elapsed_ms(started),
            )

        dense_hits: list[SearchHit] = []
        sparse_hits: list[SearchHit] = []
        errors: list[str] = []
        reranked = False
        model_id = self._embedder.model_id

        if mode in {"dense", "hybrid", "hybrid_rerank"}:
            with start_span("retrieval.embed", provider=self._settings.embeddings_provider) as span:
                try:
                    vector = self._embedder.embed_query(normalized)
                    model_id = self._embedder.model_id
                    span.set_attribute("model", model_id)
                    span.set_attribute("dimensions", len(vector))
                except Exception as error:
                    span.record_exception(error)
                    vector = []
                    errors.append(f"Embedding failed: {type(error).__name__}: {error}")
            if vector:
                with start_span("retrieval.dense", model=model_id) as span:
                    result = self._executor.run(self._dense_sql(vector, model_id))
                    if result.ok:
                        dense_hits = _hits(result, rank_kind="dense")
                        span.set_attribute("rows_returned", len(dense_hits))
                    else:
                        errors.append(result.error or "Dense retrieval failed")

        if mode in {"sparse", "hybrid", "hybrid_rerank"}:
            with start_span("retrieval.sparse", model=model_id) as span:
                result = self._executor.run(self._sparse_sql(normalized))
                if result.ok:
                    sparse_hits = _hits(result, rank_kind="sparse")
                    span.set_attribute("rows_returned", len(sparse_hits))
                else:
                    errors.append(result.error or "Sparse retrieval failed")

        if mode == "dense":
            hits = dense_hits[:limit]
        elif mode == "sparse":
            hits = sparse_hits[:limit]
        else:
            with start_span("retrieval.fuse", rrf_k=self._settings.retrieval_rrf_k):
                hits = rrf_fuse(
                    dense_hits,
                    sparse_hits,
                    rrf_k=self._settings.retrieval_rrf_k,
                )
            if mode == "hybrid_rerank" and hits:
                if self._reranker is None:
                    errors.append("Reranker is not configured or could not be loaded")
                else:
                    with start_span("retrieval.rerank", candidates=len(hits)) as span:
                        try:
                            scores = self._reranker.scores(
                                normalized,
                                [hit.content for hit in hits],
                            )
                            hits = sorted(
                                (
                                    replace(hit, score=float(score))
                                    for hit, score in zip(hits, scores, strict=True)
                                ),
                                key=lambda hit: (-hit.score, hit.citation),
                            )
                            reranked = True
                        except Exception as error:
                            span.record_exception(error)
                            errors.append(f"Reranking failed: {type(error).__name__}: {error}")
            hits = hits[:limit]

        fatal = not hits and errors
        return RetrievalResult(
            query=normalized,
            mode=mode,
            hits=tuple(hits),
            latency_ms=_elapsed_ms(started),
            error="; ".join(errors)[:2000] if fatal else None,
            warnings=tuple(errors) if not fatal else (),
            reranked=reranked,
        )

    @property
    def embedding_model(self) -> str:
        return self._embedder.model_id

    def _dense_sql(self, vector: list[float], model_id: str) -> str:
        vector_literal = "[" + ",".join(_finite_float(value) for value in vector) + "]"
        candidates = self._settings.retrieval_dense_candidates
        model = _sql_literal(model_id)
        return f"""
            SELECT doc_id, chunk_id, content, source_path, heading,
                   1 - (embedding <=> '{vector_literal}'::vector) AS score
            FROM embeddings
            WHERE embedding_model = {model}
            ORDER BY embedding <=> '{vector_literal}'::vector
            LIMIT {candidates}
        """

    def _sparse_sql(self, query: str) -> str:
        candidates = self._settings.retrieval_sparse_candidates
        term = _sql_literal(query)
        # websearch_to_tsquery ANDs every term, so a long natural-language
        # question matches almost nothing in a small corpus and the sparse ranker
        # contributes zero to the fusion (hybrid collapses to dense). Rewrite the
        # parsed query to OR semantics so lexical matches on any term surface.
        tsquery = f"replace(websearch_to_tsquery('english', {term})::text, ' & ', ' | ')::tsquery"
        return f"""
            SELECT doc_id, chunk_id, content, source_path, heading,
                   MAX(ts_rank_cd(tsv, {tsquery})) AS score
            FROM embeddings
            WHERE tsv @@ {tsquery}
            GROUP BY doc_id, chunk_id, content, source_path, heading
            ORDER BY score DESC, doc_id, chunk_id
            LIMIT {candidates}
        """


def rrf_fuse(
    dense_hits: list[SearchHit],
    sparse_hits: list[SearchHit],
    *,
    rrf_k: int = 60,
) -> list[SearchHit]:
    """Fuse two rankings deterministically using Reciprocal Rank Fusion."""

    by_citation: dict[str, SearchHit] = {}
    scores: dict[str, float] = {}
    dense_ranks = {hit.citation: rank for rank, hit in enumerate(dense_hits, 1)}
    sparse_ranks = {hit.citation: rank for rank, hit in enumerate(sparse_hits, 1)}
    for hit in [*dense_hits, *sparse_hits]:
        by_citation.setdefault(hit.citation, hit)
    for citation in by_citation:
        score = 0.0
        if citation in dense_ranks:
            score += 1.0 / (rrf_k + dense_ranks[citation])
        if citation in sparse_ranks:
            score += 1.0 / (rrf_k + sparse_ranks[citation])
        scores[citation] = score
    return sorted(
        (
            replace(
                hit,
                score=scores[citation],
                dense_rank=dense_ranks.get(citation),
                sparse_rank=sparse_ranks.get(citation),
            )
            for citation, hit in by_citation.items()
        ),
        key=lambda hit: (-hit.score, hit.citation),
    )


def _hits(result: ExecutionResult, *, rank_kind: str) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for rank, row in enumerate(result.rows, 1):
        hits.append(
            SearchHit(
                doc_id=str(row["doc_id"]),
                chunk_id=str(row["chunk_id"]),
                content=str(row["content"]),
                source_path=str(row["source_path"]),
                heading=str(row["heading"]) if row.get("heading") is not None else None,
                score=float(row.get("score") or 0.0),
                dense_rank=rank if rank_kind == "dense" else None,
                sparse_rank=rank if rank_kind == "sparse" else None,
            )
        )
    return hits


def _sql_literal(value: str) -> str:
    return "'" + value.replace("\x00", "").replace("'", "''") + "'"


def _finite_float(value: float) -> str:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError("Embedding contains a non-finite value")
    return format(resolved, ".9g")


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


_shared_retrievers: dict[tuple[object, ...], HybridRetriever] = {}
_shared_lock = threading.Lock()


def get_shared_retriever(settings: Settings | None = None) -> HybridRetriever:
    """Reuse the lazy ONNX model across API requests in one process."""

    resolved = settings or get_settings()
    key = (
        resolved.read_only_database_url,
        resolved.embeddings_provider,
        resolved.fastembed_model,
        resolved.openai_embedding_model,
        resolved.embedding_dimensions,
        resolved.reranker_provider,
    )
    with _shared_lock:
        retriever = _shared_retrievers.get(key)
        if retriever is None:
            retriever = HybridRetriever(resolved)
            _shared_retrievers[key] = retriever
        return retriever
