"""Small, explicit Postgres access layer shared by API components."""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.config import Settings, get_settings
from observability.tracing import start_span


def connect_writer(settings: Settings | None = None) -> psycopg.Connection[dict[str, Any]]:
    """Open a short-lived writer connection used by the indexer."""

    resolved = settings or get_settings()
    return psycopg.connect(resolved.database_url, row_factory=dict_row, autocommit=True)


def connect_reader(settings: Settings | None = None) -> psycopg.Connection[dict[str, Any]]:
    """Open a short-lived connection authenticated as the SELECT-only role."""

    resolved = settings or get_settings()
    return psycopg.connect(resolved.read_only_database_url, row_factory=dict_row, autocommit=True)


def freshness_snapshot(settings: Settings | None = None) -> dict[str, Any]:
    """Return the latest successful data watermark and most recent sync state."""

    with start_span("db.execute", operation="data_freshness") as span:
        with connect_reader(settings) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT token_address, chain, name, symbol, last_synced_at
                FROM tokens
                ORDER BY last_synced_at DESC
                LIMIT 1
                """
            )
            token = cursor.fetchone()
            cursor.execute(
                """
                SELECT id, status, started_at, finished_at, rows_fetched,
                       rows_upserted, rows_deleted, transfer_pages_fetched,
                       transfers_fetched, transfers_upserted, error
                FROM sync_runs
                ORDER BY id DESC
                LIMIT 1
                """
            )
            latest_run = cursor.fetchone()
        span.set_attribute("rows_returned", int(token is not None) + int(latest_run is not None))
    return {"token": token, "latest_sync": latest_run}
