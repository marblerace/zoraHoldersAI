"""Pure MCP tool implementations, separated from transport registration for testing."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Protocol

from app.config import Settings, get_settings
from db.core import freshness_snapshot
from db.schema_context import SchemaSnapshot, introspect_schema
from observability.tracing import start_span
from sql_guard.executor import ExecutionResult, SQLExecutor


class Executor(Protocol):
    def run(self, query: str) -> ExecutionResult: ...


class MCPTools:
    """Four MCP operations sharing the production guard and read-only executor."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        executor: Executor | None = None,
        schema_loader: Callable[[Settings], SchemaSnapshot] = introspect_schema,
        freshness_loader: Callable[[Settings], dict[str, Any]] = freshness_snapshot,
    ) -> None:
        self._settings = settings or get_settings()
        self._executor = executor or SQLExecutor(self._settings)
        self._schema_loader = schema_loader
        self._freshness_loader = freshness_loader

    def run_sql(self, query: str) -> dict[str, Any]:
        """Execute one guarded read-only SELECT and always return structured output."""

        with (
            start_span("mcp.call", tool="run_sql"),
            start_span("tool.run_sql") as span,
        ):
            if not isinstance(query, str) or not query.strip():
                return {
                    "columns": [],
                    "rows": [],
                    "row_count": 0,
                    "blocked": True,
                    "reason": "run_sql requires a non-empty query",
                }
            result = self._executor.run(query)
            span.set_attribute(
                "guard_decision",
                "allowed" if result.guard.allowed else "blocked",
            )
            span.set_attribute("rows_returned", len(result.rows))
            span.set_attribute("latency_ms", result.latency_ms)
            return _execution_payload(result)

    def describe_schema(self) -> dict[str, Any]:
        """Return the same runtime schema context supplied to the agent."""

        with start_span("mcp.call", tool="describe_schema"):
            snapshot = self._schema_loader(self._settings)
            return {
                "schema": snapshot.schema_text,
                "last_synced_at": (
                    snapshot.last_synced_at.isoformat() if snapshot.last_synced_at else None
                ),
            }

    def data_freshness(self) -> dict[str, Any]:
        """Return the tracked token watermark and latest indexer run."""

        with start_span("mcp.call", tool="data_freshness"):
            return _json_value(self._freshness_loader(self._settings))

    def top_holders(self, limit: int = 10) -> dict[str, Any]:
        """Return a bounded current-holder ranking through the same SQL executor."""

        with start_span("mcp.call", tool="top_holders"):
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
                return {
                    "columns": [],
                    "rows": [],
                    "row_count": 0,
                    "blocked": True,
                    "reason": "limit must be an integer between 1 and 100",
                }
            token = self._settings.tracked_token_address
            query = (
                "SELECT holder_address, balance, balance_decimal "
                "FROM holders "
                f"WHERE token_address = '{token}' "
                f"ORDER BY balance DESC LIMIT {limit}"
            )
            with start_span("tool.run_sql") as span:
                result = self._executor.run(query)
                span.set_attribute(
                    "guard_decision",
                    "allowed" if result.guard.allowed else "blocked",
                )
                span.set_attribute("rows_returned", len(result.rows))
                span.set_attribute("latency_ms", result.latency_ms)
                return _execution_payload(result)


def _execution_payload(result: ExecutionResult) -> dict[str, Any]:
    if result.ok:
        return {
            "columns": list(result.columns),
            "rows": list(result.rows),
            "row_count": len(result.rows),
            "blocked": False,
            "reason": None,
            "latency_ms": result.latency_ms,
        }
    blocked = not result.guard.allowed
    return {
        "columns": [],
        "rows": [],
        "row_count": 0,
        "blocked": blocked,
        "reason": result.guard.reason if blocked else result.error,
        "error": result.error,
        "latency_ms": result.latency_ms,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value
