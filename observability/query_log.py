"""Persist and emit one structured record for every agent request."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from decimal import Decimal

import psycopg

from app.config import Settings
from db.core import connect_writer
from llm.types import TokenUsage

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QueryLogRecord:
    question: str
    provider: str
    model: str
    status: str
    final_sql: str | None
    usage: TokenUsage
    cost_usd: Decimal | None
    latency_ms: int
    retries: int
    guard_rejection: str | None = None
    error: str | None = None
    reason: str | None = None
    served_from_cache: bool = False
    rows_returned: int = 0


def record_query(record: QueryLogRecord, settings: Settings) -> None:
    """Log to stdout and best-effort insert into query_logs."""

    stdout_payload = {
        "event": "agent_query",
        **asdict(record),
        "usage": record.usage.to_dict(),
    }
    logger.info(json.dumps(stdout_payload, default=str, ensure_ascii=False))

    try:
        with connect_writer(settings) as connection:
            connection.execute(
                """
                INSERT INTO query_logs (
                    question, provider, model, status, final_sql,
                    tokens_in, tokens_out, cache_read_tokens, cache_write_tokens,
                    cost_usd, latency_ms, retries, guard_rejection, error,
                    reason, served_from_cache, rows_returned
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s
                )
                """,
                (
                    record.question,
                    record.provider,
                    record.model,
                    record.status,
                    record.final_sql,
                    record.usage.input_tokens,
                    record.usage.output_tokens,
                    record.usage.cache_read_input_tokens,
                    record.usage.cache_write_input_tokens,
                    record.cost_usd,
                    record.latency_ms,
                    record.retries,
                    record.guard_rejection,
                    record.error,
                    record.reason,
                    record.served_from_cache,
                    record.rows_returned,
                ),
            )
    except psycopg.Error:
        logger.exception("query_log_database_write_failed")
