"""Small serializable types shared by retrieval, the agent, and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    doc_id: str
    chunk_id: str
    content: str
    source_path: str
    heading: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.doc_id}#{self.chunk_id}"


@dataclass(frozen=True, slots=True)
class SearchHit:
    doc_id: str
    chunk_id: str
    content: str
    source_path: str
    heading: str | None
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None

    @property
    def citation(self) -> str:
        return f"{self.doc_id}#{self.chunk_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation": self.citation,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source_path": self.source_path,
            "heading": self.heading,
            "score": self.score,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
        }


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    mode: str
    hits: tuple[SearchHit, ...] = ()
    latency_ms: int = 0
    error: str | None = None
    warnings: tuple[str, ...] = ()
    reranked: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def citations(self) -> tuple[str, ...]:
        return tuple(hit.citation for hit in self.hits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "query": self.query,
            "mode": self.mode,
            "hits": [hit.to_dict() for hit in self.hits],
            "citations": list(self.citations),
            "latency_ms": self.latency_ms,
            "error": self.error,
            "warnings": list(self.warnings),
            "reranked": self.reranked,
        }

    def tool_payload(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False)
