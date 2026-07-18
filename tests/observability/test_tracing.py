from __future__ import annotations

from app.config import Settings
from observability import tracing


def test_tracer_is_noop_without_langfuse_configuration(monkeypatch) -> None:
    monkeypatch.setattr(
        tracing,
        "get_settings",
        lambda: Settings(
            _env_file=None,
            langfuse_public_key=None,
            langfuse_secret_key=None,
            langfuse_host=None,
        ),
    )
    tracing.reset_tracing_for_tests()

    with tracing.start_span("test.span", value=1) as span:
        span.set_attribute("nullable", None)

    assert tracing.tracing_enabled() is False
