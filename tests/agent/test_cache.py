from __future__ import annotations

from datetime import UTC, datetime, timedelta

from agent.cache import MemoryAnswerCache, normalize_question
from agent.resilience import CircuitBreaker
from agent.service import TextToSQLAgent
from app.config import Settings
from db.schema_context import SchemaSnapshot
from llm.client import LLMProviderError
from llm.types import Completion, Message, TokenUsage, ToolCall, ToolDefinition
from sql_guard.executor import ExecutionResult
from sql_guard.guard import GuardResult


class FakeLLM:
    provider = "openai"
    model = "gpt-5.6-terra"

    def __init__(self, completions: list[Completion], *, error: Exception | None = None) -> None:
        self.completions = completions
        self.error = error
        self.calls = 0

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> Completion:
        del messages, tools
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.completions.pop(0)


class FakeExecutor:
    def run(self, query: str) -> ExecutionResult:
        return ExecutionResult(
            ok=True,
            guard=GuardResult(True, "allowed", f"{query} LIMIT 1000"),
            columns=("holder_count",),
            rows=({"holder_count": 10},),
        )


def _completion(*, query: str | None = None, text: str = "") -> Completion:
    calls = (
        (ToolCall(id="sql", name="run_sql", arguments={"query": query}),)
        if query is not None
        else ()
    )
    return Completion(text=text, tool_calls=calls, usage=TokenUsage(10, 2))


def _agent(
    llm: FakeLLM,
    cache: MemoryAnswerCache,
    settings: Settings,
) -> TextToSQLAgent:
    snapshot = SchemaSnapshot("CREATE TABLE holders (holder_count bigint);", None)
    return TextToSQLAgent(
        settings,
        llm=llm,
        executor=FakeExecutor(),
        cache=cache,
        circuit=CircuitBreaker(100, 30),
        schema_loader=lambda _: snapshot,
        query_logger=lambda *_: None,
    )


def test_normalized_question_collapses_case_punctuation_and_whitespace() -> None:
    assert normalize_question("  HOW many... holders?! ") == "how many holders"


def test_fresh_cache_hit_skips_provider() -> None:
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    cache = MemoryAnswerCache(clock=lambda: now[0])
    settings = Settings(
        _env_file=None,
        enable_scheduler=False,
        answer_cache_ttl_seconds=60,
    )
    first_llm = FakeLLM(
        [
            _completion(query="SELECT COUNT(*) AS holder_count FROM holders"),
            _completion(text="There are 10 holders."),
        ]
    )
    first = _agent(first_llm, cache, settings).ask("How many holders?")
    unused_llm = FakeLLM([])

    second = _agent(unused_llm, cache, settings).ask("HOW many holders!!!")

    assert first.status == "succeeded"
    assert second.status == "succeeded"
    assert second.served_from_cache is True
    assert second.token_usage.total_tokens == 0
    assert unused_llm.calls == 0


def test_stale_cache_is_served_when_provider_is_unavailable() -> None:
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    cache = MemoryAnswerCache(clock=lambda: now[0])
    settings = Settings(
        _env_file=None,
        enable_scheduler=False,
        answer_cache_ttl_seconds=1,
        llm_provider_retry_attempts=0,
    )
    healthy = FakeLLM(
        [
            _completion(query="SELECT COUNT(*) AS holder_count FROM holders"),
            _completion(text="There are 10 holders."),
        ]
    )
    assert _agent(healthy, cache, settings).ask("How many holders?").status == "succeeded"
    now[0] += timedelta(seconds=2)
    failing = FakeLLM([], error=LLMProviderError("connection unavailable"))

    result = _agent(failing, cache, settings).ask("How many holders?")

    assert result.status == "degraded"
    assert result.reason == "provider_unavailable"
    assert result.served_from_cache is True
    assert result.answer == "There are 10 holders."
