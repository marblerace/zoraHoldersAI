"""Bounded, observable tool loop for grounded SQL and document answers."""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

from agent.cache import (
    AnswerCache,
    CacheRecord,
    DatabaseAnswerCache,
    NullAnswerCache,
    hash_schema,
    make_cache_key,
    normalize_question,
)
from agent.prompt import RUN_SQL_TOOL, SEARCH_DOCS_TOOL, build_system_prompt
from agent.resilience import (
    CircuitBreaker,
    ProviderUnavailableError,
    call_with_resilience,
    get_circuit_breaker,
)
from app.config import Settings, get_settings
from db.schema_context import SchemaSnapshot, introspect_schema
from llm.client import LLMConfigurationError, create_llm_client
from llm.pricing import estimate_cost_usd
from llm.types import Completion, LLMClient, Message, TokenUsage, ToolCall
from observability.query_log import QueryLogRecord, record_query
from observability.tracing import Span, start_span
from retrieval.models import RetrievalResult
from sql_guard.executor import ExecutionResult, SQLExecutor

logger = logging.getLogger(__name__)
_CITATION = re.compile(r"\[([A-Za-z0-9_.-]+#[A-Za-z0-9_.-]+)\]")


class Executor(Protocol):
    def run(self, query: str) -> ExecutionResult: ...


class Retriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        mode: str = "hybrid",
    ) -> RetrievalResult: ...


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Complete API-facing result for one natural-language question."""

    answer: str
    sql: str | None
    columns: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    token_usage: TokenUsage
    cost_usd: Decimal | None
    latency_ms: int
    retries: int
    provider: str
    model: str
    data_as_of: str | None
    status: str
    guard_rejection: str | None = None
    error: str | None = None
    reason: str | None = None
    last_error: str | None = None
    served_from_cache: bool = False
    citations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sql": self.sql,
            "columns": self.columns,
            "rows": self.rows,
            "token_usage": self.token_usage.to_dict(),
            "cost_usd": float(self.cost_usd) if self.cost_usd is not None else None,
            "latency_ms": self.latency_ms,
            "retries": self.retries,
            "provider": self.provider,
            "model": self.model,
            "data_as_of": self.data_as_of,
            "status": self.status,
            "guard_rejection": self.guard_rejection,
            "error": self.error,
            "reason": self.reason,
            "last_error": self.last_error,
            "served_from_cache": self.served_from_cache,
            "citations": self.citations,
        }


@dataclass(slots=True)
class _RunState:
    usage: TokenUsage
    final_sql: str | None = None
    columns: tuple[str, ...] = ()
    rows: tuple[dict[str, Any], ...] = ()
    provider_retries: int = 0
    sql_corrections: int = 0
    tool_retries: int = 0
    tool_calls: int = 0
    guard_rejection: str | None = None
    last_error: str | None = None
    citations: tuple[str, ...] = ()

    @property
    def retries(self) -> int:
        return self.provider_retries + self.sql_corrections + self.tool_retries


@dataclass(frozen=True, slots=True)
class _ToolOutcome:
    call: ToolCall
    ok: bool
    payload: str
    sql: ExecutionResult | None = None
    retrieval: RetrievalResult | None = None


class TextToSQLAgent:
    """Use at most two grounded tool calls, including one SQL correction."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLMClient | None = None,
        executor: Executor | None = None,
        retriever: Retriever | None = None,
        cache: AnswerCache | None = None,
        circuit: CircuitBreaker | None = None,
        schema_loader: Callable[[Settings], SchemaSnapshot] = introspect_schema,
        query_logger: Callable[[QueryLogRecord, Settings], None] = record_query,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = llm
        self._executor = executor or SQLExecutor(self._settings)
        self._retriever = retriever
        self._cache = cache or (
            DatabaseAnswerCache(self._settings)
            if self._settings.answer_cache_enabled
            else NullAnswerCache()
        )
        circuit_key = f"{self._provider}:{self._model}"
        self._circuit = circuit or get_circuit_breaker(
            circuit_key,
            failure_threshold=self._settings.llm_circuit_failure_threshold,
            reset_seconds=self._settings.llm_circuit_reset_seconds,
        )
        self._schema_loader = schema_loader
        self._query_logger = query_logger

    def ask(self, question: str) -> AgentResult:
        """Answer with grounded tool evidence or return a machine-readable degraded result."""

        started = time.perf_counter()
        state = _RunState(usage=TokenUsage())
        snapshot: SchemaSnapshot | None = None
        cache_key: str | None = None
        configuration_error = _provider_configuration_error(self._settings)
        with start_span("agent.ask", provider=self._provider, model=self._model) as trace:
            try:
                snapshot = self._schema_loader(self._settings)
                schema_digest = hash_schema(snapshot.schema_text)
                cache_key = make_cache_key(
                    question,
                    self._settings.tracked_token_address,
                    schema_digest,
                )
                cached = self._cache.get(cache_key)
                if cached is not None:
                    result = self._result_from_cache(
                        cached,
                        snapshot=snapshot,
                        started=started,
                        degraded=False,
                    )
                    self._observe_result(trace, result)
                    self._write_result_log(question, result)
                    return result

                messages = [
                    Message(
                        role="system",
                        content=build_system_prompt(snapshot, self._settings),
                    ),
                    Message(role="user", content=question),
                ]
                first = self._complete(messages, with_tools=True, state=state)
                messages.append(first.as_assistant_message())

                if not first.tool_calls:
                    result = self._finish(
                        question,
                        state,
                        snapshot,
                        started,
                        answer="Please clarify the metric or time window so I can query it safely.",
                        status="clarification",
                    )
                    self._observe_result(trace, result)
                    return result

                outcomes = self._execute_calls(first.tool_calls, state)
                self._append_tool_outputs(messages, first.tool_calls, outcomes)

                if not any(outcome.ok for outcome in outcomes) and state.tool_calls < 2:
                    correction = self._complete(messages, with_tools=True, state=state)
                    messages.append(correction.as_assistant_message())
                    if correction.tool_calls:
                        if outcomes and outcomes[-1].sql is not None:
                            state.sql_corrections += 1
                        corrected = self._execute_calls(correction.tool_calls, state)
                        outcomes.extend(corrected)
                        self._append_tool_outputs(messages, correction.tool_calls, corrected)

                if not any(outcome.ok for outcome in outcomes):
                    result = self._finish(
                        question,
                        state,
                        snapshot,
                        started,
                        answer="I couldn't answer that confidently.",
                        status="degraded",
                        reason=(
                            "guard_rejected"
                            if state.guard_rejection is not None
                            else "no_valid_sql"
                        ),
                    )
                    self._observe_result(trace, result)
                    return result

                final = self._complete(messages, with_tools=False, state=state)
                answer = final.text or self._fallback_grounded_answer(outcomes)
                answer = _enforce_returned_citations(answer, state.citations)
                result = self._finish(
                    question,
                    state,
                    snapshot,
                    started,
                    answer=answer,
                    status="succeeded",
                )
                if cache_key is not None:
                    self._cache.put(
                        cache_key,
                        normalized_question=normalize_question(question),
                        token_address=self._settings.tracked_token_address,
                        schema_hash=schema_digest,
                        payload=result.to_dict(),
                        ttl_seconds=self._settings.answer_cache_ttl_seconds,
                    )
                self._observe_result(trace, result)
                return result
            except (ProviderUnavailableError, LLMConfigurationError) as error:
                state.last_error = f"{type(error).__name__}: {error}"[:2000]
                retries = getattr(error, "retries", 0)
                state.provider_retries += int(retries)
                if cache_key is not None and snapshot is not None:
                    stale = self._cache.get(cache_key, allow_stale=True)
                    if stale is not None:
                        result = self._result_from_cache(
                            stale,
                            snapshot=snapshot,
                            started=started,
                            degraded=True,
                            last_error=state.last_error,
                        )
                        self._observe_result(trace, result)
                        self._write_result_log(question, result)
                        return result
                result = self._finish(
                    question,
                    state,
                    snapshot,
                    started,
                    answer="I couldn't answer that confidently.",
                    status="degraded",
                    reason="provider_unavailable",
                )
                self._observe_result(trace, result)
                return result
            except Exception as error:
                if configuration_error:
                    state.last_error = f"LLMConfigurationError: {configuration_error}"[:2000]
                    reason = "provider_unavailable"
                else:
                    state.last_error = f"{type(error).__name__}: {error}"[:2000]
                    reason = "no_valid_sql"
                trace.record_exception(error)
                result = self._finish(
                    question,
                    state,
                    snapshot,
                    started,
                    answer="I couldn't answer that confidently.",
                    status="degraded",
                    reason=reason,
                )
                self._observe_result(trace, result)
                return result

    @property
    def _provider(self) -> str:
        return self._llm.provider if self._llm is not None else self._settings.llm_provider

    @property
    def _model(self) -> str:
        return self._llm.model if self._llm is not None else self._settings.selected_llm_model

    def _ensure_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = create_llm_client(self._settings)
        return self._llm

    def _complete(
        self,
        messages: list[Message],
        *,
        with_tools: bool,
        state: _RunState,
    ) -> Completion:
        tools = [RUN_SQL_TOOL, SEARCH_DOCS_TOOL] if with_tools else []
        started = time.perf_counter()
        with start_span("llm.generate", provider=self._provider, model=self._model) as span:
            completion, retries = call_with_resilience(
                lambda: self._ensure_llm().complete(messages, tools),
                circuit=self._circuit,
                max_retries=self._settings.llm_provider_retry_attempts,
                backoff_seconds=self._settings.llm_provider_backoff_seconds,
            )
            state.provider_retries += retries
            state.usage = state.usage + completion.usage
            span.set_attribute("prompt_tokens", completion.usage.input_tokens)
            span.set_attribute("completion_tokens", completion.usage.output_tokens)
            span.set_attribute("retries", retries)
            span.set_attribute("latency_ms", _elapsed_ms(started))
            span.set_attribute(
                "cost_usd",
                estimate_cost_usd(self._model, completion.usage, self._settings),
            )
            return completion

    def _execute_calls(
        self,
        calls: tuple[ToolCall, ...],
        state: _RunState,
    ) -> list[_ToolOutcome]:
        outcomes: list[_ToolOutcome] = []
        for call in calls:
            if state.tool_calls >= 2:
                outcomes.append(
                    _ToolOutcome(
                        call=call,
                        ok=False,
                        payload=_error_payload("The two-tool-call limit has been reached."),
                    )
                )
                continue
            state.tool_calls += 1
            outcome = self._execute_call(call, state)
            outcomes.append(outcome)
            if outcome.sql is not None and outcome.sql.transient and state.tool_calls < 2:
                delay = self._settings.llm_provider_backoff_seconds
                if delay:
                    time.sleep(delay)
                state.tool_calls += 1
                state.tool_retries += 1
                outcome = self._execute_call(call, state)
                outcomes[-1] = outcome
        return outcomes

    def _execute_call(self, call: ToolCall, state: _RunState) -> _ToolOutcome:
        query = call.arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            message = f"{call.name} requires a non-empty string query"
            state.last_error = message
            return _ToolOutcome(call=call, ok=False, payload=_error_payload(message))

        if call.name == RUN_SQL_TOOL.name:
            with start_span("tool.run_sql") as span:
                execution = self._executor.run(query)
                self._capture_sql(state, execution)
                span.set_attribute(
                    "guard_decision",
                    "allowed" if execution.guard.allowed else "blocked",
                )
                span.set_attribute("rows_returned", len(execution.rows))
                span.set_attribute("latency_ms", execution.latency_ms)
                return _ToolOutcome(
                    call=call,
                    ok=execution.ok,
                    payload=execution.tool_payload(),
                    sql=execution,
                )

        if call.name == SEARCH_DOCS_TOOL.name:
            with start_span("tool.search_docs") as span:
                if not self._settings.retrieval_enabled:
                    result = RetrievalResult(
                        query=query,
                        mode="hybrid",
                        error="Document retrieval is disabled",
                    )
                else:
                    result = self._ensure_retriever().search(query, mode="hybrid")
                state.last_error = result.error
                if result.ok:
                    state.citations = tuple(dict.fromkeys((*state.citations, *result.citations)))
                span.set_attribute("rows_returned", len(result.hits))
                span.set_attribute("latency_ms", result.latency_ms)
                return _ToolOutcome(
                    call=call,
                    ok=result.ok,
                    payload=result.tool_payload(),
                    retrieval=result,
                )

        message = f"Unknown tool: {call.name}"
        state.last_error = message
        return _ToolOutcome(call=call, ok=False, payload=_error_payload(message))

    def _ensure_retriever(self) -> Retriever:
        if self._retriever is None:
            from retrieval.service import get_shared_retriever

            self._retriever = get_shared_retriever(self._settings)
        return self._retriever

    @staticmethod
    def _append_tool_outputs(
        messages: list[Message],
        calls: tuple[ToolCall, ...],
        outcomes: list[_ToolOutcome],
    ) -> None:
        by_id = {outcome.call.id: outcome for outcome in outcomes}
        for call in calls:
            outcome = by_id.get(call.id)
            if outcome is None:
                payload = _error_payload("Tool call was not executed")
                is_error = True
            else:
                payload = outcome.payload
                is_error = not outcome.ok
            messages.append(
                Message(
                    role="tool",
                    tool_call_id=call.id,
                    content=payload,
                    is_error=is_error,
                )
            )

    @staticmethod
    def _capture_sql(state: _RunState, execution: ExecutionResult) -> None:
        state.last_error = execution.error
        if execution.guard.allowed:
            state.final_sql = execution.guard.safe_sql
        else:
            state.guard_rejection = execution.guard.reason
            state.final_sql = None
        state.columns = execution.columns
        state.rows = execution.rows

    def _finish(
        self,
        question: str,
        state: _RunState,
        snapshot: SchemaSnapshot | None,
        started: float,
        *,
        answer: str,
        status: str,
        reason: str | None = None,
    ) -> AgentResult:
        result = AgentResult(
            answer=answer.strip(),
            sql=state.final_sql,
            columns=state.columns,
            rows=state.rows,
            token_usage=state.usage,
            cost_usd=estimate_cost_usd(self._model, state.usage, self._settings),
            latency_ms=_elapsed_ms(started),
            retries=state.retries,
            provider=self._provider,
            model=self._model,
            data_as_of=(
                snapshot.last_synced_at.isoformat()
                if snapshot is not None and snapshot.last_synced_at
                else None
            ),
            status=status,
            guard_rejection=state.guard_rejection,
            error=state.last_error if status == "degraded" else None,
            reason=reason,
            last_error=state.last_error if status == "degraded" else None,
            citations=state.citations,
        )
        self._write_result_log(question, result)
        return result

    def _result_from_cache(
        self,
        record: CacheRecord,
        *,
        snapshot: SchemaSnapshot,
        started: float,
        degraded: bool,
        last_error: str | None = None,
    ) -> AgentResult:
        payload = record.payload
        previous_cost = payload.get("cost_usd")
        cost = None if previous_cost is None else Decimal("0")
        return AgentResult(
            answer=str(payload.get("answer") or "I couldn't answer that confidently."),
            sql=payload.get("sql"),
            columns=tuple(payload.get("columns") or ()),
            rows=tuple(dict(row) for row in (payload.get("rows") or ())),
            token_usage=TokenUsage(),
            cost_usd=cost,
            latency_ms=_elapsed_ms(started),
            retries=0,
            provider=str(payload.get("provider") or self._provider),
            model=str(payload.get("model") or self._model),
            data_as_of=(
                snapshot.last_synced_at.isoformat()
                if snapshot.last_synced_at
                else payload.get("data_as_of")
            ),
            status="degraded" if degraded else "succeeded",
            guard_rejection=payload.get("guard_rejection"),
            error=last_error if degraded else None,
            reason="provider_unavailable" if degraded else None,
            last_error=last_error if degraded else None,
            served_from_cache=True,
            citations=tuple(payload.get("citations") or ()),
        )

    def _write_result_log(self, question: str, result: AgentResult) -> None:
        self._write_log(
            QueryLogRecord(
                question=question,
                provider=result.provider,
                model=result.model,
                status=result.status,
                final_sql=result.sql,
                usage=result.token_usage,
                cost_usd=result.cost_usd,
                latency_ms=result.latency_ms,
                retries=result.retries,
                guard_rejection=result.guard_rejection,
                error=result.last_error,
                reason=result.reason,
                served_from_cache=result.served_from_cache,
                rows_returned=len(result.rows),
            )
        )

    def _write_log(self, record: QueryLogRecord) -> None:
        try:
            self._query_logger(record, self._settings)
        except Exception:
            logger.exception("query_logger_failed")

    @staticmethod
    def _observe_result(trace: Span, result: AgentResult) -> None:
        trace.set_attribute("status", result.status)
        trace.set_attribute("reason", result.reason)
        trace.set_attribute("prompt_tokens", result.token_usage.input_tokens)
        trace.set_attribute("completion_tokens", result.token_usage.output_tokens)
        trace.set_attribute("cost_usd", result.cost_usd)
        trace.set_attribute("latency_ms", result.latency_ms)
        trace.set_attribute("retries", result.retries)
        trace.set_attribute("guard_decision", result.guard_rejection or "allowed")
        trace.set_attribute("rows_returned", len(result.rows))
        trace.set_attribute("served_from_cache", result.served_from_cache)

    @staticmethod
    def _fallback_grounded_answer(outcomes: list[_ToolOutcome]) -> str:
        sql = next(
            (outcome.sql for outcome in reversed(outcomes) if outcome.sql and outcome.ok),
            None,
        )
        if sql is not None:
            if not sql.rows:
                return "The query returned no matching rows."
            if len(sql.rows) == 1 and len(sql.columns) == 1:
                value = sql.rows[0][sql.columns[0]]
                return f"The query returned {value}."
            return f"The query returned {len(sql.rows)} rows; the preview is included below."
        retrieval = next(
            (
                outcome.retrieval
                for outcome in reversed(outcomes)
                if outcome.retrieval and outcome.ok
            ),
            None,
        )
        if retrieval is not None and retrieval.citations:
            cited = ", ".join(f"[{citation}]" for citation in retrieval.citations)
            return f"I found relevant project documentation in {cited}."
        return "I couldn't answer that confidently."


def _error_payload(reason: str) -> str:
    return json.dumps({"ok": False, "error": reason}, separators=(",", ":"))


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _provider_configuration_error(settings: Settings) -> str | None:
    if settings.llm_provider == "anthropic":
        secret = settings.anthropic_api_key
        if secret is None or not secret.get_secret_value().strip():
            return "LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY for POST /ask"
    elif settings.llm_provider == "openai":
        secret = settings.openai_api_key
        if secret is None or not secret.get_secret_value().strip():
            return "LLM_PROVIDER=openai requires OPENAI_API_KEY for POST /ask"
    elif shutil.which(settings.claude_code_command) is None:
        return "LLM_PROVIDER=claude_code requires the authenticated local Claude Code CLI"
    return None


def _enforce_returned_citations(answer: str, citations: tuple[str, ...]) -> str:
    """Remove fabricated chunk IDs and ensure retrieved evidence remains visible."""

    if not citations:
        return answer
    allowed = set(citations)
    grounded = _CITATION.sub(
        lambda match: match.group(0) if match.group(1) in allowed else "",
        answer,
    ).strip()
    if not any(f"[{citation}]" in grounded for citation in citations):
        sources = ", ".join(f"[{citation}]" for citation in citations)
        grounded = f"{grounded}\n\nSources: {sources}".strip()
    return grounded
