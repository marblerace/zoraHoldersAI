"""Provider-neutral embeddings with a local FastEmbed default and offline fallback."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Any, Protocol

from app.config import Settings, get_settings
from llm.client import LLMConfigurationError, LLMProviderError

logger = logging.getLogger(__name__)
_WORD = re.compile(r"[a-z0-9]+")


class EmbeddingBackend(Protocol):
    dimension: int

    @property
    def model_id(self) -> str: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashEmbeddingBackend:
    """Dependency-free, deterministic fallback for fully air-gapped startup.

    This is intentionally a fallback rather than the benchmark backend.  It keeps
    local/no-key mode functional when the FastEmbed ONNX model has not yet been
    cached on disk.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    @property
    def model_id(self) -> str:
        return f"local-hash-v1-{self.dimension}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        values = [0.0] * self.dimension
        for token in _WORD.findall(text.casefold()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            values[index] += sign
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]


class FastEmbedBackend:
    """Lazy ONNX BGE embeddings; falls back locally if package/model is unavailable."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self.dimension = self._settings.embedding_dimensions
        self._model: Any | None = None
        self._fallback: HashEmbeddingBackend | None = None

    @property
    def model_id(self) -> str:
        if self._fallback is not None:
            return self._fallback.model_id
        return self._settings.fastembed_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        if self._fallback is not None:
            return self._fallback.embed_documents(texts)
        method = getattr(model, "passage_embed", model.embed)
        return [_float_vector(vector, self.dimension) for vector in method(texts)]

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        if self._fallback is not None:
            return self._fallback.embed_query(text)
        method = getattr(model, "query_embed", model.embed)
        return _float_vector(next(iter(method([text]))), self.dimension)

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self._fallback is not None:
            return self._fallback
        try:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(
                model_name=self._settings.fastembed_model,
                cache_dir=str(self._settings.fastembed_cache_dir),
                local_files_only=self._settings.fastembed_local_files_only,
            )
        except Exception as error:
            logger.warning(
                "fastembed_unavailable fallback=local_hash error=%s",
                f"{type(error).__name__}: {error}",
            )
            self._fallback = HashEmbeddingBackend(self.dimension)
            return self._fallback
        return self._model


class OpenAIEmbeddingBackend:
    """Optional paid embedding backend, selected explicitly by configuration."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        secret = self._settings.openai_api_key
        key = secret.get_secret_value().strip() if secret else ""
        if not key:
            raise LLMConfigurationError("EMBEDDINGS_PROVIDER=openai requires OPENAI_API_KEY")
        from openai import OpenAI

        self._client = OpenAI(api_key=key, timeout=self._settings.llm_timeout_seconds)
        self.dimension = self._settings.embedding_dimensions

    @property
    def model_id(self) -> str:
        return self._settings.openai_embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(
                model=self._settings.openai_embedding_model,
                input=texts,
                dimensions=self.dimension,
            )
        except Exception as error:
            raise LLMProviderError(f"OpenAI embeddings failed: {error}") from error
        return [_float_vector(item.embedding, self.dimension) for item in response.data]


class Reranker(Protocol):
    def scores(self, query: str, documents: list[str]) -> list[float]: ...


class FastEmbedReranker:
    def __init__(self, settings: Settings | None = None) -> None:
        resolved = settings or get_settings()
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(
            model_name=resolved.fastembed_reranker_model,
            cache_dir=str(resolved.fastembed_cache_dir),
            local_files_only=resolved.fastembed_local_files_only,
        )

    def scores(self, query: str, documents: list[str]) -> list[float]:
        return [float(value) for value in self._model.rerank(query, documents)]


def create_embedding_backend(settings: Settings | None = None) -> EmbeddingBackend:
    resolved = settings or get_settings()
    if resolved.embeddings_provider == "openai":
        return OpenAIEmbeddingBackend(resolved)
    return FastEmbedBackend(resolved)


def _float_vector(vector: Any, expected_dimension: int) -> list[float]:
    values = [float(value) for value in vector]
    if len(values) != expected_dimension:
        raise ValueError(
            f"Embedding dimension {len(values)} does not match configured "
            f"dimension {expected_dimension}"
        )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Embedding contains a non-finite value")
    return values
