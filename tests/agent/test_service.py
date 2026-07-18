from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from agent.cache import NullAnswerCache
from agent.service import TextToSQLAgent
from app.config import Settings
from db.schema_context import SchemaSnapshot
from llm.types import Completion, Message, TokenUsage, ToolCall, ToolDefinition
from retrieval.models import RetrievalResult, SearchHit
from sql_guard.executor import ExecutionResult
from sql_guard.guard import GuardResult


class FakeLLM:
    provider = "openai"
    model = "gpt-5.6-terra"

    def __init__(self, completions: list[Completion]) -> None:
        self.completions = completions
        self.calls: list[tuple[list[Message], list[ToolDefinition]]] = []

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> Completion:
        self.calls.append((list(messages), list(tools)))
        return self.completions.pop(0)


class FakeExecutor:
    def __init__(self, results: list[ExecutionResult]) -> None:
        self.results = results
        self.queries: list[str] = []

    def run(self, query: str) -> ExecutionResult:
        self.queries.append(query)
        return self.results.pop(0)


class FakeRetriever:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
        self.queries: list[str] = []

    def search(self, query: str, *, top_k=None, mode="hybrid") -> RetrievalResult:
        del top_k, mode
        self.queries.append(query)
        return self.result


def completion(
    *,
    text: str = "",
    query: str | None = None,
    call_id: str = "call-1",
    usage: TokenUsage | None = None,
) -> Completion:
    calls = ()
    if query is not None:
        calls = (ToolCall(id=call_id, name="run_sql", arguments={"query": query}),)
    resolved_usage = usage or TokenUsage(input_tokens=100, output_tokens=20)
    return Completion(text=text, tool_calls=calls, usage=resolved_usage)


def execution(
    *,
    ok: bool,
    safe_sql: str = "",
    rows: tuple[dict[str, object], ...] = (),
    error: str | None = None,
    rejection: str | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        ok=ok,
        guard=GuardResult(
            allowed=not bool(rejection),
            reason=rejection or "allowed",
            safe_sql=safe_sql,
        ),
        columns=tuple(rows[0]) if rows else (),
        rows=rows,
        error=error,
    )


def build_agent(llm: FakeLLM, executor: FakeExecutor, logs: list) -> TextToSQLAgent:
    settings = Settings(_env_file=None, enable_scheduler=False)
    snapshot = SchemaSnapshot(
        schema_text="CREATE TABLE holders (holder_address text, balance numeric);",
        last_synced_at=datetime(2026, 7, 16, 8, 30, tzinfo=UTC),
    )
    return TextToSQLAgent(
        settings,
        llm=llm,
        executor=executor,
        cache=NullAnswerCache(),
        schema_loader=lambda _: snapshot,
        query_logger=lambda record, _: logs.append(record),
    )


def test_successful_query_is_executed_once_then_summarized() -> None:
    llm = FakeLLM(
        [
            completion(query="SELECT COUNT(*) AS holder_count FROM holders"),
            completion(
                text="There are 1,774 current holders.",
                usage=TokenUsage(input_tokens=50, output_tokens=10),
            ),
        ]
    )
    executor = FakeExecutor(
        [
            execution(
                ok=True,
                safe_sql="SELECT COUNT(*) AS holder_count FROM holders LIMIT 1000",
                rows=({"holder_count": 1774},),
            )
        ]
    )
    logs: list = []

    result = build_agent(llm, executor, logs).ask("How many holders are there?")

    assert result.answer == "There are 1,774 current holders."
    assert result.rows == ({"holder_count": 1774},)
    assert result.retries == 0
    assert result.status == "succeeded"
    assert result.token_usage.input_tokens == 150
    assert result.token_usage.output_tokens == 30
    assert result.cost_usd == Decimal("0.00082500")
    assert executor.queries == ["SELECT COUNT(*) AS holder_count FROM holders"]
    assert [len(tools) for _, tools in llm.calls] == [2, 0]
    assert len(logs) == 1


def test_guard_error_gets_exactly_one_correction_attempt() -> None:
    llm = FakeLLM(
        [
            completion(query="DELETE FROM holders", call_id="bad"),
            completion(
                query="SELECT COUNT(*) AS holder_count FROM holders",
                call_id="fixed",
            ),
            completion(text="There are 1,774 holders."),
        ]
    )
    executor = FakeExecutor(
        [
            execution(ok=False, error="write blocked", rejection="Only SELECT queries"),
            execution(
                ok=True,
                safe_sql="SELECT COUNT(*) AS holder_count FROM holders LIMIT 1000",
                rows=({"holder_count": 1774},),
            ),
        ]
    )
    logs: list = []

    result = build_agent(llm, executor, logs).ask("Count holders")

    assert result.status == "succeeded"
    assert result.retries == 1
    assert result.guard_rejection == "Only SELECT queries"
    assert executor.queries == [
        "DELETE FROM holders",
        "SELECT COUNT(*) AS holder_count FROM holders",
    ]
    assert [len(tools) for _, tools in llm.calls] == [2, 2, 0]


def test_ambiguous_question_can_return_clarification_without_sql() -> None:
    llm = FakeLLM([completion(text="Which time window should I use?")])
    executor = FakeExecutor([])
    logs: list = []

    result = build_agent(llm, executor, logs).ask("How many recent holders?")

    assert result.answer == "Please clarify the metric or time window so I can query it safely."
    assert result.status == "clarification"
    assert result.sql is None
    assert result.rows == ()
    assert executor.queries == []
    assert len(llm.calls) == 1


def test_direct_answer_without_tool_is_replaced_by_safe_clarification() -> None:
    llm = FakeLLM([completion(text="There are definitely 9,999 holders.")])
    executor = FakeExecutor([])
    logs: list = []

    result = build_agent(llm, executor, logs).ask("How many holders?")

    assert result.status == "clarification"
    assert result.answer == "Please clarify the metric or time window so I can query it safely."
    assert "9,999" not in result.answer


def test_second_tool_failure_stops_without_a_third_sql_attempt() -> None:
    llm = FakeLLM(
        [
            completion(query="SELECT missing FROM holders", call_id="first"),
            completion(query="SELECT also_missing FROM holders", call_id="second"),
            completion(text="I couldn't resolve that query from the schema."),
        ]
    )
    executor = FakeExecutor(
        [
            execution(
                ok=False,
                safe_sql="SELECT missing FROM holders LIMIT 1000",
                error="column missing does not exist",
            ),
            execution(
                ok=False,
                safe_sql="SELECT also_missing FROM holders LIMIT 1000",
                error="column also_missing does not exist",
            ),
        ]
    )
    logs: list = []

    result = build_agent(llm, executor, logs).ask("Use a missing column")

    assert result.status == "degraded"
    assert result.reason == "no_valid_sql"
    assert result.retries == 1
    assert result.last_error == "column also_missing does not exist"
    assert len(executor.queries) == 2
    assert [len(tools) for _, tools in llm.calls] == [2, 2]


def test_search_docs_path_returns_and_exposes_citations() -> None:
    llm = FakeLLM(
        [
            Completion(
                text="",
                tool_calls=(
                    ToolCall(
                        id="docs",
                        name="search_docs",
                        arguments={"query": "meaning of first seen"},
                    ),
                ),
                usage=TokenUsage(20, 5),
            ),
            completion(
                text=(
                    "It is the indexer's first observation, not proven acquisition time "
                    "[methodology#meaning-of-first-seen] [invented#source]."
                )
            ),
        ]
    )
    retriever = FakeRetriever(
        RetrievalResult(
            query="meaning of first seen",
            mode="hybrid",
            hits=(
                SearchHit(
                    "methodology",
                    "meaning-of-first-seen",
                    "First observed by the indexer.",
                    "retrieval/corpus/methodology.md",
                    "Meaning of first seen",
                    0.9,
                ),
            ),
        )
    )
    settings = Settings(_env_file=None, enable_scheduler=False)
    snapshot = SchemaSnapshot("CREATE TABLE holders (holder_address text);", None)
    agent = TextToSQLAgent(
        settings,
        llm=llm,
        executor=FakeExecutor([]),
        retriever=retriever,
        cache=NullAnswerCache(),
        schema_loader=lambda _: snapshot,
        query_logger=lambda *_: None,
    )

    result = agent.ask("What does first_seen_at mean?")

    assert result.status == "succeeded"
    assert result.sql is None
    assert result.citations == ("methodology#meaning-of-first-seen",)
    assert "[methodology#meaning-of-first-seen]" in result.answer
    assert "invented#source" not in result.answer
    assert retriever.queries == ["meaning of first seen"]
