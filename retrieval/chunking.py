"""Deterministic, heading-aware Markdown chunking with bounded token overlap."""

from __future__ import annotations

import re
from pathlib import Path

from retrieval.models import DocumentChunk

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_NON_SLUG = re.compile(r"[^a-z0-9]+")


def load_corpus(
    corpus_dir: Path,
    *,
    max_tokens: int = 220,
    overlap_tokens: int = 40,
) -> list[DocumentChunk]:
    """Load every Markdown document in stable path order."""

    chunks: list[DocumentChunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        chunks.extend(
            chunk_markdown(
                path.read_text(encoding="utf-8"),
                doc_id=path.stem,
                source_path=str(path),
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
        )
    return chunks


def chunk_markdown(
    markdown: str,
    *,
    doc_id: str,
    source_path: str,
    max_tokens: int = 220,
    overlap_tokens: int = 40,
) -> list[DocumentChunk]:
    """Split by headings first, then by approximate tokens when a section is long."""

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be between zero and max_tokens - 1")

    sections: list[tuple[str, list[str]]] = []
    heading = "overview"
    body: list[str] = []
    for line in markdown.splitlines():
        match = _HEADING.match(line)
        if match:
            if any(part.strip() for part in body):
                sections.append((heading, body))
            heading = match.group(2).strip()
            body = []
        else:
            body.append(line)
    if any(part.strip() for part in body):
        sections.append((heading, body))

    chunks: list[DocumentChunk] = []
    slug_counts: dict[str, int] = {}
    for section_heading, lines in sections:
        text = "\n".join(lines).strip()
        if not text:
            continue
        base_slug = _slug(section_heading)
        duplicate = slug_counts.get(base_slug, 0)
        slug_counts[base_slug] = duplicate + 1
        if duplicate:
            base_slug = f"{base_slug}-{duplicate + 1}"
        tokens = text.split()
        step = max_tokens - overlap_tokens
        for part, start in enumerate(range(0, len(tokens), step), 1):
            window = tokens[start : start + max_tokens]
            if not window:
                break
            chunk_id = base_slug if len(tokens) <= max_tokens else f"{base_slug}-{part:02d}"
            chunks.append(
                DocumentChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    heading=section_heading,
                    content=" ".join(window),
                    source_path=source_path,
                    metadata={"token_count": len(window)},
                )
            )
            if start + max_tokens >= len(tokens):
                break
    return chunks


def _slug(value: str) -> str:
    slug = _NON_SLUG.sub("-", value.casefold()).strip("-")
    return slug or "section"
