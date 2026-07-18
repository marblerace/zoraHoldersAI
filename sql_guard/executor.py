"""Execute guard-approved SQL inside a short read-only Postgres transaction."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import psycopg

from app.config import Settings, get_settings
from db.core import connect_reader
from observability.tracing import start_span
from sql_guard.guard import GuardResult, SQLGuard

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Serializable SQL execution outcome returned to the agent tool loop."""

    ok: bool
    guard: GuardResult
    columns: tuple[str, ...] = ()
    rows: tuple[dict[str, Any], ...] = ()
    error: str | None = None
    latency_ms: int = 0
    transient: bool = False

    def tool_payload(self) -> str:
        """Return compact JSON suitable for a tool-result message."""

        payload = {
            "ok": self.ok,
            "columns": self.columns,
            "rows": self.rows,
            "error": self.error,
        }
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


class SQLExecutor:
    """Validate and execute analytics queries with database-enforced bounds."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        guard: SQLGuard | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._guard = guard or SQLGuard(max_rows=self._settings.sql_max_rows)

    def run(self, query: str) -> ExecutionResult:
        """Return rows or a structured rejection/error without raising to the LLM loop."""

        started = time.perf_counter()
        decision = self._guard.validate(query)
        if not decision.allowed:
            return ExecutionResult(
                ok=False,
                guard=decision,
                error=f"SQL guard rejected the query: {decision.reason}",
                latency_ms=_elapsed_ms(started),
            )

        with start_span(
            "db.execute",
            statement_timeout_ms=self._settings.sql_statement_timeout_ms,
        ) as span:
            try:
                with (
                    connect_reader(self._settings) as connection,
                    connection.transaction(),
                    connection.cursor() as cursor,
                ):
                    timeout_ms = self._settings.sql_statement_timeout_ms
                    cursor.execute("SET TRANSACTION READ ONLY")
                    cursor.execute(f"SET LOCAL statement_timeout = '{timeout_ms}ms'")
                    cursor.execute(decision.safe_sql)
                    columns = tuple(column.name for column in (cursor.description or ()))
                    rows = tuple(
                        {key: _to_json_value(value) for key, value in dict(database_row).items()}
                        for database_row in cursor.fetchall()
                    )
            except psycopg.errors.QueryCanceled as error:
                message = (
                    "Query exceeded the "
                    f"{self._settings.sql_statement_timeout_ms}ms statement timeout"
                )
                logger.warning("sql_execution_timeout sql=%s", decision.safe_sql)
                span.record_exception(error)
                span.set_attribute("latency_ms", _elapsed_ms(started))
                return ExecutionResult(
                    ok=False,
                    guard=decision,
                    error=message,
                    latency_ms=_elapsed_ms(started),
                )
            except psycopg.Error as error:
                primary = error.diag.message_primary or type(error).__name__
                message = f"Database rejected the query: {primary}"[:2000]
                transient = _is_transient_database_error(error)
                logger.warning("sql_execution_error error=%s sql=%s", primary, decision.safe_sql)
                span.record_exception(error)
                span.set_attribute("transient", transient)
                span.set_attribute("latency_ms", _elapsed_ms(started))
                return ExecutionResult(
                    ok=False,
                    guard=decision,
                    error=message,
                    latency_ms=_elapsed_ms(started),
                    transient=transient,
                )

            span.set_attribute("rows_returned", len(rows))
            span.set_attribute("latency_ms", _elapsed_ms(started))

        return ExecutionResult(
            ok=True,
            guard=decision,
            columns=columns,
            rows=rows,
            latency_ms=_elapsed_ms(started),
        )


def _to_json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, bytes):
        return f"0x{value.hex()}"
    return value


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _is_transient_database_error(error: psycopg.Error) -> bool:
    sqlstate = error.sqlstate or ""
    return isinstance(error, psycopg.OperationalError) or sqlstate.startswith(
        ("08", "53", "57P", "58")
    )
