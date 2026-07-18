from __future__ import annotations

import pytest

from agent.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    call_with_resilience,
)


def test_circuit_opens_after_threshold_and_resets_after_window() -> None:
    now = [100.0]
    circuit = CircuitBreaker(2, 30, clock=lambda: now[0])

    circuit.before_call()
    circuit.record_failure()
    circuit.before_call()
    circuit.record_failure()

    with pytest.raises(CircuitOpenError):
        circuit.before_call()
    assert circuit.snapshot().state == "open"

    now[0] += 31
    circuit.before_call()
    assert circuit.snapshot().state == "half_open"
    circuit.record_success()
    assert circuit.snapshot().state == "closed"
    circuit.before_call()


def test_transient_provider_call_retries_without_opening_healthy_circuit() -> None:
    attempts = [0]
    circuit = CircuitBreaker(4, 30)

    def operation() -> str:
        attempts[0] += 1
        if attempts[0] < 3:
            raise TimeoutError("temporary timeout")
        return "ok"

    result, retries = call_with_resilience(
        operation,
        circuit=circuit,
        max_retries=2,
        backoff_seconds=0,
    )

    assert result == "ok"
    assert retries == 2
    assert attempts[0] == 3
    assert circuit.snapshot().state == "closed"
