from __future__ import annotations

from pathlib import Path

from app.config import Settings
from retrieval.chunking import load_corpus
from retrieval.embeddings import HashEmbeddingBackend
from retrieval.models import SearchHit
from retrieval.service import HybridRetriever, rrf_fuse
from sql_guard.executor import ExecutionResult
from sql_guard.guard import SQLGuard


class FakeExecutor:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.guard = SQLGuard(max_rows=1000)

    def run(self, query: str) -> ExecutionResult:
        self.queries.append(query)
        decision = self.guard.validate(query)
        assert decision.allowed, decision.reason
        sparse = "TS_RANK" in decision.safe_sql
        rows = (
            {
                "doc_id": "onchain_glossary" if sparse else "methodology",
                "chunk_id": "mint" if sparse else "acquisition-timing",
                "content": "A grounded test chunk.",
                "source_path": "retrieval/corpus/test.md",
                "heading": "Test",
                "score": 0.8 if sparse else 0.7,
            },
        )
        return ExecutionResult(
            ok=True,
            guard=decision,
            columns=tuple(rows[0]),
            rows=rows,
        )


def _hit(citation: str, score: float) -> SearchHit:
    doc_id, chunk_id = citation.split("#")
    return SearchHit(doc_id, chunk_id, "content", "source.md", None, score)


def test_corpus_chunks_have_stable_provenance() -> None:
    corpus = Path(__file__).parents[2] / "retrieval" / "corpus"
    chunks = load_corpus(corpus)

    assert len(chunks) >= 20
    assert len({chunk.citation for chunk in chunks}) == len(chunks)
    assert all("#" in chunk.citation and chunk.content for chunk in chunks)


def test_hybrid_retrieval_returns_dense_and_sparse_citations_through_guard() -> None:
    executor = FakeExecutor()
    settings = Settings(_env_file=None, enable_scheduler=False)
    retriever = HybridRetriever(
        settings,
        embedder=HashEmbeddingBackend(384),
        executor=executor,
    )

    result = retriever.search("What is a mint?", top_k=5, mode="hybrid")

    assert result.ok is True
    assert result.citations == (
        "methodology#acquisition-timing",
        "onchain_glossary#mint",
    )
    assert len(executor.queries) == 2
    assert all(executor.guard.validate(query).allowed for query in executor.queries)


def test_empty_query_is_handled_without_sql() -> None:
    executor = FakeExecutor()
    retriever = HybridRetriever(
        Settings(_env_file=None),
        embedder=HashEmbeddingBackend(),
        executor=executor,
    )

    result = retriever.search("   ")

    assert result.ok is False
    assert "non-empty" in result.error
    assert executor.queries == []


def test_zero_top_k_and_unknown_mode_are_rejected() -> None:
    executor = FakeExecutor()
    retriever = HybridRetriever(
        Settings(_env_file=None),
        embedder=HashEmbeddingBackend(),
        executor=executor,
    )

    zero = retriever.search("mint", top_k=0)
    unknown = retriever.search("mint", mode="magic")

    assert zero.ok is False
    assert "between 1 and 20" in zero.error
    assert unknown.ok is False
    assert "Unsupported" in unknown.error
    assert executor.queries == []


def test_rrf_rewards_chunks_present_in_both_rankings() -> None:
    dense = [_hit("docs#a", 0.9), _hit("docs#b", 0.8)]
    sparse = [_hit("docs#b", 4.0), _hit("docs#c", 3.0)]

    fused = rrf_fuse(dense, sparse, rrf_k=60)

    assert fused[0].citation == "docs#b"
    assert fused[0].dense_rank == 2
    assert fused[0].sparse_rank == 1


def test_dependency_free_embedding_fallback_needs_no_network() -> None:
    backend = HashEmbeddingBackend(384)

    documents = backend.embed_documents(["mint from the zero address"])
    query = backend.embed_query("mint")

    assert len(documents[0]) == 384
    assert len(query) == 384
    assert documents[0] == backend.embed_documents(["mint from the zero address"])[0]
