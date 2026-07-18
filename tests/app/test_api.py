from decimal import Decimal

from fastapi.testclient import TestClient

import app.main as app_main
from agent.service import AgentResult
from app.config import Settings, get_settings
from app.main import app
from llm.types import TokenUsage


def test_ask_requires_configured_provider_key() -> None:
    settings = Settings(
        _env_file=None,
        enable_scheduler=False,
        llm_provider="anthropic",
        anthropic_api_key=None,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        response = TestClient(app).post("/ask", json={"question": "Count holders"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["reason"] == "provider_unavailable"
    assert "ANTHROPIC_API_KEY" in response.json()["last_error"]


def test_ask_rejects_whitespace_only_question() -> None:
    response = TestClient(app).post("/ask", json={"question": "   "})

    assert response.status_code == 422


def test_admin_sync_requires_token_before_touching_database() -> None:
    settings = Settings(_env_file=None, enable_scheduler=False, admin_sync_token="secret")
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        response = TestClient(app).post("/admin/sync")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 401


def test_guard_degradation_is_http_200(monkeypatch) -> None:
    result = AgentResult(
        answer="I couldn't answer that confidently.",
        sql=None,
        columns=(),
        rows=(),
        token_usage=TokenUsage(),
        cost_usd=Decimal("0"),
        latency_ms=2,
        retries=1,
        provider="openai",
        model="test-model",
        data_as_of=None,
        status="degraded",
        guard_rejection="Only SELECT queries are allowed",
        error="query blocked",
        reason="guard_rejected",
        last_error="query blocked",
    )

    class StubAgent:
        def __init__(self, settings) -> None:
            del settings

        def ask(self, question: str) -> AgentResult:
            del question
            return result

    monkeypatch.setattr(app_main, "TextToSQLAgent", StubAgent)
    settings = Settings(_env_file=None, enable_scheduler=False)
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        response = TestClient(app).post("/ask", json={"question": "DROP everything"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["reason"] == "guard_rejected"
