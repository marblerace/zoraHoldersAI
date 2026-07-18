"""Normalized answer cache used to avoid repeat model calls and serve stale fallback."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

import psycopg

from app.config import Settings, get_settings
from db.core import connect_writer
from observability.tracing import start_span

logger = logging.getLogger(__name__)
_PUNCTUATION = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class CacheRecord:
    key: str
    payload: dict[str, Any]
    created_at: datetime
    expires_at: datetime

    @property
    def stale(self) -> bool:
        return self.expires_at <= datetime.now(UTC)


class AnswerCache(Protocol):
    def get(self, key: str, *, allow_stale: bool = False) -> CacheRecord | None: ...

    def put(
        self,
        key: str,
        *,
        normalized_question: str,
        token_address: str,
        schema_hash: str,
        payload: dict[str, Any],
        ttl_seconds: int,
    ) -> None: ...


class CacheMetrics:
    """Process-local cache counters exposed from the health endpoint."""

    def __init__(self) -> None:
        self._hits = 0
        self._misses = 0
        self._stale_hits = 0
        self._lock = threading.Lock()

    def hit(self, *, stale: bool = False) -> None:
        with self._lock:
            self._hits += 1
            self._stale_hits += int(stale)

    def miss(self) -> None:
        with self._lock:
            self._misses += 1

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            lookups = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "stale_hits": self._stale_hits,
                "lookups": lookups,
                "hit_rate": self._hits / lookups if lookups else 0.0,
            }

    def reset(self) -> None:
        with self._lock:
            self._hits = self._misses = self._stale_hits = 0


cache_metrics = CacheMetrics()


class NullAnswerCache:
    """Cache implementation used when persistence is disabled."""

    def get(self, key: str, *, allow_stale: bool = False) -> CacheRecord | None:
        del key, allow_stale
        return None

    def put(self, key: str, **kwargs: Any) -> None:
        del key, kwargs


class DatabaseAnswerCache:
    """Postgres-backed cache with fixed, application-owned SQL statements."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def get(self, key: str, *, allow_stale: bool = False) -> CacheRecord | None:
        with start_span("db.execute", operation="answer_cache.get") as span:
            try:
                with connect_writer(self._settings) as connection:
                    row = connection.execute(
                        """
                        SELECT cache_key, response_json, created_at, expires_at
                        FROM answer_cache
                        WHERE cache_key = %s
                          AND (%s OR expires_at > NOW())
                        """,
                        (key, allow_stale),
                    ).fetchone()
                    if row is not None:
                        connection.execute(
                            """
                            UPDATE answer_cache
                            SET last_accessed_at = NOW(), hit_count = hit_count + 1
                            WHERE cache_key = %s
                            """,
                            (key,),
                        )
            except psycopg.Error as error:
                span.record_exception(error)
                logger.warning("answer_cache_read_failed", exc_info=True)
                return None
            span.set_attribute("rows_returned", int(row is not None))
        if row is None:
            cache_metrics.miss()
            return None
        payload = row["response_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        record = CacheRecord(
            key=row["cache_key"],
            payload=dict(payload),
            created_at=row["created_at"],
            expires_at=row["expires_at"],
        )
        cache_metrics.hit(stale=record.stale)
        return record

    def put(
        self,
        key: str,
        *,
        normalized_question: str,
        token_address: str,
        schema_hash: str,
        payload: dict[str, Any],
        ttl_seconds: int,
    ) -> None:
        with start_span("db.execute", operation="answer_cache.put") as span:
            try:
                with connect_writer(self._settings) as connection:
                    connection.execute(
                        """
                        INSERT INTO answer_cache (
                            cache_key, normalized_question, token_address, schema_hash,
                            response_json, created_at, expires_at, last_accessed_at
                        )
                        VALUES (
                            %s, %s, %s, %s, %s::jsonb, NOW(),
                            NOW() + (%s * INTERVAL '1 second'), NOW()
                        )
                        ON CONFLICT (cache_key) DO UPDATE SET
                            response_json = EXCLUDED.response_json,
                            created_at = EXCLUDED.created_at,
                            expires_at = EXCLUDED.expires_at,
                            last_accessed_at = EXCLUDED.last_accessed_at
                        """,
                        (
                            key,
                            normalized_question,
                            token_address,
                            schema_hash,
                            json.dumps(payload, default=str, ensure_ascii=False),
                            ttl_seconds,
                        ),
                    )
            except psycopg.Error as error:
                span.record_exception(error)
                logger.warning("answer_cache_write_failed", exc_info=True)
            else:
                span.set_attribute("rows_returned", 1)


class MemoryAnswerCache:
    """Deterministic cache for unit tests and small embedded deployments."""

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))
        self._records: dict[str, CacheRecord] = {}

    def get(self, key: str, *, allow_stale: bool = False) -> CacheRecord | None:
        record = self._records.get(key)
        if record is None or (record.expires_at <= self._clock() and not allow_stale):
            cache_metrics.miss()
            return None
        cache_metrics.hit(stale=record.expires_at <= self._clock())
        return record

    def put(
        self,
        key: str,
        *,
        normalized_question: str,
        token_address: str,
        schema_hash: str,
        payload: dict[str, Any],
        ttl_seconds: int,
    ) -> None:
        del normalized_question, token_address, schema_hash
        now = self._clock()
        self._records[key] = CacheRecord(
            key=key,
            payload=payload,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )


def normalize_question(question: str) -> str:
    """Case-fold and collapse punctuation/whitespace for stable cache identity."""

    without_punctuation = _PUNCTUATION.sub(" ", question.casefold())
    return _WHITESPACE.sub(" ", without_punctuation).strip()


def hash_schema(schema_text: str) -> str:
    return hashlib.sha256(schema_text.encode("utf-8")).hexdigest()


def make_cache_key(question: str, token_address: str, schema_hash: str) -> str:
    normalized = normalize_question(question)
    identity = "\n".join((normalized, token_address.casefold(), schema_hash))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()
