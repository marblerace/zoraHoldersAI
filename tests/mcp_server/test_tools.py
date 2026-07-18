from __future__ import annotations

from datetime import UTC, datetime

from app.config import Settings
from db.schema_context import SchemaSnapshot
from mcp_server.tools import MCPTools
from sql_guard.executor import ExecutionResult
from sql_guard.guard import SQLGuard


class GuardOnlyExecutor:
    def __init__(self) -> None:
        self.guard = SQLGuard(max_rows=1000)
        self.queries: list[str] = []

    def run(self, query: str) -> ExecutionResult:
        self.queries.append(query)
        decision = self.guard.validate(query)
        if not decision.allowed:
            return ExecutionResult(
                ok=False,
                guard=decision,
                error=f"SQL guard rejected the query: {decision.reason}",
            )
        rows = (
            {
                "holder_address": "0x1111111111111111111111111111111111111111",
                "balance": "100",
                "balance_decimal": "100",
            },
        )
        return ExecutionResult(
            ok=True,
            guard=decision,
            columns=tuple(rows[0]),
            rows=rows,
        )


def _tools(executor: GuardOnlyExecutor | None = None) -> MCPTools:
    settings = Settings(_env_file=None, enable_scheduler=False)
    snapshot = SchemaSnapshot(
        "CREATE TABLE holders (holder_address text, balance numeric);",
        datetime(2026, 7, 17, tzinfo=UTC),
    )
    return MCPTools(
        settings,
        executor=executor or GuardOnlyExecutor(),
        schema_loader=lambda _: snapshot,
        freshness_loader=lambda _: {
            "token": {"last_synced_at": datetime(2026, 7, 17, tzinfo=UTC)},
            "latest_sync": {"id": 7, "status": "succeeded"},
        },
    )


def test_run_sql_blocks_attacks_with_structured_error() -> None:
    result = _tools().run_sql("SELECT pg_read_file('/etc/passwd')")

    assert result["blocked"] is True
    assert result["row_count"] == 0
    assert "forbidden" in result["reason"].lower()


def test_run_sql_returns_serializable_shape() -> None:
    result = _tools().run_sql("SELECT holder_address, balance FROM holders")

    assert result["blocked"] is False
    assert result["row_count"] == 1
    assert result["columns"] == ["holder_address", "balance", "balance_decimal"]


def test_describe_schema_and_freshness_have_stable_shapes() -> None:
    tools = _tools()

    schema = tools.describe_schema()
    freshness = tools.data_freshness()

    assert schema["schema"].startswith("CREATE TABLE holders")
    assert schema["last_synced_at"].startswith("2026-07-17")
    assert freshness["token"]["last_synced_at"].startswith("2026-07-17")
    assert freshness["latest_sync"] == {"id": 7, "status": "succeeded"}


def test_top_holders_uses_injected_guarded_executor() -> None:
    executor = GuardOnlyExecutor()
    result = _tools(executor).top_holders(5)

    assert result["blocked"] is False
    assert len(executor.queries) == 1
    assert "LIMIT 5" in executor.queries[0]
    assert executor.guard.validate(executor.queries[0]).allowed is True


def test_top_holders_rejects_out_of_range_limit_without_database_call() -> None:
    executor = GuardOnlyExecutor()

    result = _tools(executor).top_holders(1000)

    assert result["blocked"] is True
    assert executor.queries == []
