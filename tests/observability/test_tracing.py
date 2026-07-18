from __future__ import annotations

import sys
from types import SimpleNamespace

from opentelemetry.sdk.resources import SERVICE_NAME

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


def test_langfuse_client_uses_named_otel_resource(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeLangfuse:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "langfuse", SimpleNamespace(Langfuse=FakeLangfuse))
    tracing.reset_tracing_for_tests()

    client = tracing._build_client(
        "pk-test",
        "sk-test",
        "http://langfuse.test",
        "zora-analytics-agent",
    )

    assert isinstance(client, FakeLangfuse)
    provider = captured["tracer_provider"]
    assert provider.resource.attributes[SERVICE_NAME] == "zora-analytics-agent"
    tracing.reset_tracing_for_tests()
