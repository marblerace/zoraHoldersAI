"""Keep tests isolated from developer-local telemetry configuration."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.config import get_settings
from observability.tracing import reset_tracing_for_tests


@pytest.fixture(autouse=True)
def disable_external_tracing(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Prevent a local ``.env`` from exporting test spans to Langfuse."""

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    monkeypatch.setenv("LANGFUSE_HOST", "")
    get_settings.cache_clear()
    reset_tracing_for_tests()
    yield
    get_settings.cache_clear()
    reset_tracing_for_tests()
