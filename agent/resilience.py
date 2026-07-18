"""Bounded provider retries and a process-wide circuit breaker."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import httpx

from llm.client import LLMProviderError

T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    """Raised before a provider call while its circuit is open."""


class ProviderUnavailableError(RuntimeError):
    """Raised after a provider call exhausts its bounded retry policy."""

    def __init__(self, message: str, *, retries: int = 0) -> None:
        super().__init__(message)
        self.retries = retries


@dataclass(frozen=True, slots=True)
class CircuitSnapshot:
    state: str
    consecutive_failures: int
    retry_after_seconds: float | None


class CircuitBreaker:
    """Thread-safe closed/open/half-open provider circuit breaker."""

    def __init__(
        self,
        failure_threshold: int = 4,
        reset_seconds: float = 30.0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be positive")
        if reset_seconds <= 0:
            raise ValueError("reset_seconds must be positive")
        self._threshold = failure_threshold
        self._reset_seconds = reset_seconds
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._half_open_probe = False
        self._lock = threading.Lock()

    def before_call(self) -> None:
        """Allow a call, a single reset probe, or raise while still open."""

        with self._lock:
            if self._opened_at is None:
                return
            elapsed = self._clock() - self._opened_at
            if elapsed < self._reset_seconds:
                raise CircuitOpenError(
                    f"LLM provider circuit is open; retry in "
                    f"{max(0.0, self._reset_seconds - elapsed):.1f}s"
                )
            if self._half_open_probe:
                raise CircuitOpenError("LLM provider circuit is awaiting a recovery probe")
            self._half_open_probe = True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._half_open_probe = False

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._half_open_probe or self._failures >= self._threshold:
                self._opened_at = self._clock()
                self._half_open_probe = False

    def snapshot(self) -> CircuitSnapshot:
        with self._lock:
            if self._opened_at is None:
                return CircuitSnapshot("closed", self._failures, None)
            remaining = self._reset_seconds - (self._clock() - self._opened_at)
            state = "half_open" if self._half_open_probe else "open"
            return CircuitSnapshot(state, self._failures, max(0.0, remaining))


def call_with_resilience(
    operation: Callable[[], T],
    *,
    circuit: CircuitBreaker,
    max_retries: int,
    backoff_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[T, int]:
    """Run a provider operation with transient-only exponential backoff.

    The returned integer counts provider retries and is intentionally separate
    from SQL correction attempts in the agent loop.
    """

    retries = 0
    while True:
        try:
            circuit.before_call()
        except CircuitOpenError as error:
            raise ProviderUnavailableError(str(error), retries=retries) from error
        try:
            value = operation()
        except Exception as error:
            circuit.record_failure()
            if not is_transient_provider_error(error) or retries >= max_retries:
                raise ProviderUnavailableError(
                    f"{type(error).__name__}: {error}",
                    retries=retries,
                ) from error
            delay = min(30.0, backoff_seconds * (2**retries))
            retries += 1
            if delay > 0:
                sleep(delay)
            continue
        circuit.record_success()
        return value, retries


def is_transient_provider_error(error: BaseException) -> bool:
    """Conservatively classify network, timeout, throttling, and 5xx failures."""

    if isinstance(error, (TimeoutError, ConnectionError, httpx.TimeoutException)):
        return True
    if isinstance(error, httpx.TransportError):
        return True
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int) and (status_code == 429 or status_code >= 500):
        return True
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int) and (response_status == 429 or response_status >= 500):
        return True
    if isinstance(error, LLMProviderError):
        message = str(error).casefold()
        markers = (
            "timeout",
            "timed out",
            "connection",
            "temporarily",
            "unavailable",
            "rate limit",
            "429",
            "500",
            "502",
            "503",
            "504",
        )
        return any(marker in message for marker in markers)
    return False


_registry: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_circuit_breaker(
    key: str,
    *,
    failure_threshold: int,
    reset_seconds: float,
) -> CircuitBreaker:
    """Return one process-wide circuit per provider/model pair."""

    with _registry_lock:
        circuit = _registry.get(key)
        if circuit is None:
            circuit = CircuitBreaker(failure_threshold, reset_seconds)
            _registry[key] = circuit
        return circuit


def reset_circuit_registry_for_tests() -> None:
    with _registry_lock:
        _registry.clear()
