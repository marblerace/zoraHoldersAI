"""Small, fail-soft tracing facade backed by Langfuse's OpenTelemetry SDK.

Application code only depends on :func:`start_span`.  When Langfuse credentials or
the optional dependency are absent, the exact same context-manager path runs with
an in-process no-op span.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from typing import Any, Protocol

from app.config import get_settings

logger = logging.getLogger(__name__)


class Span(Protocol):
    """OpenTelemetry-shaped subset used by the application."""

    def set_attribute(self, key: str, value: Any) -> None: ...

    def record_exception(self, error: BaseException) -> None: ...


@dataclass(slots=True)
class _NoOpSpan:
    attributes: dict[str, Any] = field(default_factory=dict)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def record_exception(self, error: BaseException) -> None:
        self.attributes["error.type"] = type(error).__name__
        self.attributes["error.message"] = str(error)


@dataclass(slots=True)
class _LangfuseSpan:
    observation: Any
    attributes: dict[str, Any]

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = _json_value(value)
        self._update()

    def record_exception(self, error: BaseException) -> None:
        self.attributes["error.type"] = type(error).__name__
        self.attributes["error.message"] = str(error)[:2000]
        try:
            self.observation.update(
                level="ERROR",
                status_message=str(error)[:2000],
                metadata=self.attributes,
            )
        except Exception:
            logger.debug("langfuse_span_exception_update_failed", exc_info=True)

    def _update(self) -> None:
        try:
            self.observation.update(metadata=self.attributes)
        except Exception:
            logger.debug("langfuse_span_update_failed", exc_info=True)


@contextmanager
def start_span(name: str, **attributes: Any) -> Iterator[Span]:
    """Start a nested span or transparently yield a no-op span.

    Langfuse Python v4 is OpenTelemetry-native, so observations created here keep
    normal parent/child context semantics and can coexist with other OTel tooling.
    Instrumentation is deliberately unable to make an application request fail.
    """

    normalized = {key: _json_value(value) for key, value in attributes.items()}
    client = _get_configured_client()
    if client is None:
        yield _NoOpSpan(normalized)
        return

    observation_context: AbstractContextManager[Any] | None = None
    try:
        observation_context = client.start_as_current_observation(
            as_type="generation" if name == "llm.generate" else "span",
            name=name,
        )
        observation = observation_context.__enter__()
        span = _LangfuseSpan(observation, normalized)
        span._update()
    except Exception:
        logger.debug("langfuse_span_start_failed", exc_info=True)
        yield _NoOpSpan(normalized)
        return

    try:
        yield span
    except BaseException as error:
        span.record_exception(error)
        raise
    finally:
        try:
            observation_context.__exit__(None, None, None)
        except Exception:
            logger.debug("langfuse_span_close_failed", exc_info=True)


def tracing_enabled() -> bool:
    """Return whether a usable, credentialed Langfuse client is configured."""

    return _get_configured_client() is not None


def flush_traces() -> None:
    """Best-effort flush for short-lived CLI processes."""

    client = _get_configured_client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            logger.debug("langfuse_flush_failed", exc_info=True)


def reset_tracing_for_tests() -> None:
    """Clear the lazily created client after a test changes environment settings."""

    _build_client.cache_clear()


def _get_configured_client() -> Any | None:
    settings = get_settings()
    public_key = (settings.langfuse_public_key or "").strip()
    secret = settings.langfuse_secret_key
    secret_key = secret.get_secret_value().strip() if secret is not None else ""
    host = (settings.langfuse_host or "").rstrip("/")
    if not public_key or not secret_key or not host:
        return None
    return _build_client(public_key, secret_key, host, settings.otel_service_name)


@lru_cache(maxsize=4)
def _build_client(
    public_key: str,
    secret_key: str,
    host: str,
    service_name: str,
) -> Any | None:
    try:
        from langfuse import Langfuse
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider

        tracer_provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))

        try:
            return Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                base_url=host,
                tracer_provider=tracer_provider,
            )
        except TypeError:
            # Compatibility with Langfuse v3, which called this argument ``host``.
            return Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    except Exception:
        logger.warning("langfuse_disabled reason=initialization_failed", exc_info=True)
        return None


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)
