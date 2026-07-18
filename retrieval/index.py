"""CLI and service for indexing the curated Markdown corpus into PostgreSQL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from db.core import connect_writer
from retrieval.chunking import load_corpus
from retrieval.embeddings import EmbeddingBackend, create_embedding_backend


def index_corpus(
    settings: Settings | None = None,
    *,
    corpus_dir: Path | None = None,
    embedder: EmbeddingBackend | None = None,
) -> dict[str, Any]:
    """Replace one model's corpus index with deterministic, provenance-rich chunks."""

    resolved = settings or get_settings()
    directory = corpus_dir or Path(__file__).with_name("corpus")
    chunks = load_corpus(
        directory,
        max_tokens=resolved.retrieval_chunk_tokens,
        overlap_tokens=resolved.retrieval_chunk_overlap_tokens,
    )
    if not chunks:
        raise RuntimeError(f"No Markdown documents found in {directory}")
    backend = embedder or create_embedding_backend(resolved)
    vectors = backend.embed_documents([chunk.content for chunk in chunks])
    model_id = backend.model_id

    try:
        from pgvector import Vector
        from pgvector.psycopg import register_vector
    except ImportError as error:
        raise RuntimeError("Install the retrieval extra: pip install '.[retrieval]'") from error

    with connect_writer(resolved) as connection:
        register_vector(connection)
        with connection.transaction():
            connection.execute(
                "DELETE FROM embeddings WHERE embedding_model = %s",
                (model_id,),
            )
            for chunk, vector in zip(chunks, vectors, strict=True):
                connection.execute(
                    """
                    INSERT INTO embeddings (
                        doc_id, chunk_id, embedding_model, source_path, heading,
                        content, content_hash, metadata, embedding, indexed_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, NOW())
                    """,
                    (
                        chunk.doc_id,
                        chunk.chunk_id,
                        model_id,
                        chunk.source_path,
                        chunk.heading,
                        chunk.content,
                        hashlib.sha256(chunk.content.encode("utf-8")).hexdigest(),
                        json.dumps(chunk.metadata),
                        Vector(vector),
                    ),
                )
    return {
        "documents": len({chunk.doc_id for chunk in chunks}),
        "chunks": len(chunks),
        "embedding_model": model_id,
        "dimensions": len(vectors[0]),
        "corpus_dir": str(directory),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index the local RAG Markdown corpus")
    parser.add_argument("--corpus-dir", type=Path)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow FastEmbed to download its small ONNX model into the local cache.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    if args.allow_model_download:
        settings = settings.model_copy(update={"fastembed_local_files_only": False})
    print(json.dumps(index_corpus(settings, corpus_dir=args.corpus_dir), indent=2))


if __name__ == "__main__":
    main()
