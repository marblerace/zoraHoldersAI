from __future__ import annotations

from datetime import UTC, datetime

from agent.prompt import build_system_prompt
from app.config import Settings
from db.schema_context import SchemaSnapshot


def test_prompt_requires_minimal_projection_and_row_grounded_answers() -> None:
    prompt = build_system_prompt(
        SchemaSnapshot(
            schema_text="CREATE TABLE holders (balance_decimal numeric);",
            last_synced_at=datetime(2026, 7, 18, tzinfo=UTC),
        ),
        Settings(_env_file=None),
        now=datetime(2026, 7, 18, tzinfo=UTC),
    )

    assert "Projection is part of correctness" in prompt
    assert "SELECT one\n  aliased scalar and nothing else" in prompt
    assert "Do not append a token address" in prompt
    assert "SELECT name, symbol, token_type, chain" in prompt
    assert "SELECT last_synced_at FROM tokens" in prompt


def test_prompt_disambiguates_balance_and_acquisition_semantics() -> None:
    prompt = build_system_prompt(
        SchemaSnapshot("CREATE TABLE holders (balance_decimal numeric);", None),
        Settings(_env_file=None),
        now=datetime(2026, 7, 18, tzinfo=UTC),
    )

    assert '"have N MINT"' in prompt
    assert "current balance_decimal of N" in prompt
    assert 'undefined holder "acquisition" time is ambiguous' in prompt
    assert "Never substitute first_seen_at with a caveat" in prompt
