from __future__ import annotations

import pytest

from sql_guard.guard import SQLGuard


@pytest.fixture
def guard() -> SQLGuard:
    return SQLGuard(max_rows=1000)


@pytest.mark.parametrize(
    ("query", "expected_fragment"),
    [
        ("SELECT holder_address FROM holders", "LIMIT 1000"),
        ("SELECT * FROM public.tokens LIMIT 10", "LIMIT 10"),
        ("SELECT * FROM holders LIMIT 50000", "LIMIT 1000"),
        ("SELECT * FROM holders FETCH FIRST 50000 ROWS ONLY", "FETCH FIRST 1000"),
        (
            "WITH rich AS (SELECT * FROM holders WHERE balance > 10) SELECT * FROM rich",
            "LIMIT 1000",
        ),
        (
            "SELECT holder_address FROM holders UNION SELECT holder_address FROM holders",
            "LIMIT 1000",
        ),
        ("SELECT '; DROP TABLE holders' AS harmless", "LIMIT 1000"),
        ("/* harmless */ SELECT COUNT(*) FROM holders;", "LIMIT 1000"),
    ],
)
def test_allows_read_only_queries_with_bounded_results(
    guard: SQLGuard,
    query: str,
    expected_fragment: str,
) -> None:
    result = guard.validate(query)

    assert result.allowed is True
    assert expected_fragment in result.safe_sql


@pytest.mark.parametrize(
    ("query", "reason"),
    [
        ("", "empty"),
        ("INSERT INTO holders VALUES ('a', 'b', 1, 1, NOW(), NOW())", "Only SELECT"),
        ("UPDATE holders SET balance = 0", "Only SELECT"),
        ("DELETE FROM holders", "Only SELECT"),
        ("DROP TABLE holders", "Only SELECT"),
        ("ALTER TABLE holders ADD COLUMN owned TEXT", "Only SELECT"),
        ("TRUNCATE holders", "Only SELECT"),
        ("CREATE TABLE stolen AS SELECT * FROM holders", "Only SELECT"),
        ("COPY holders TO '/tmp/holders.csv'", "Only SELECT"),
        ("GRANT SELECT ON holders TO public", "Only SELECT"),
        ("SELECT 1; DROP TABLE holders", "Exactly one"),
        ("SELECT 1; /* comment */ DELETE FROM holders", "Exactly one"),
        (
            "WITH removed AS (DELETE FROM holders RETURNING *) SELECT * FROM removed",
            "Forbidden SQL operation",
        ),
        ("SELECT * INTO copied_holders FROM holders", "Forbidden SQL operation"),
        ("SELECT * FROM pg_tables", "System table"),
        ("SELECT * FROM pg_catalog.pg_tables", "System schema"),
        ("SELECT * FROM information_schema.tables", "System schema"),
        ("SELECT * FROM query_logs", "not allowlisted"),
        ("SELECT * FROM private.holders", "Schema is not allowlisted"),
        ("SELECT pg_sleep(10)", "Function is forbidden"),
        ("SELECT pg_read_file('/etc/passwd')", "Function is forbidden"),
        ("SELECT set_config('statement_timeout', '0', false)", "Function is forbidden"),
        ("SELECT nextval('sync_runs_id_seq')", "Function is forbidden"),
        ("SELECT * FROM holders FOR UPDATE", "Forbidden SQL operation"),
        ("SELECT * FROM holders LIMIT ALL", "LIMIT must be"),
        ("VALUES (1), (2)", "Only SELECT"),
    ],
)
def test_rejects_writes_injection_and_unsafe_reads(
    guard: SQLGuard,
    query: str,
    reason: str,
) -> None:
    result = guard.validate(query)

    assert result.allowed is False
    assert result.safe_sql == ""
    assert reason.lower() in result.reason.lower()


def test_rejects_multiple_select_statements(guard: SQLGuard) -> None:
    result = guard.validate("SELECT 1; SELECT 2")

    assert result.allowed is False
    assert "exactly one" in result.reason.lower()


def test_preserves_pgvector_cosine_operator_for_embeddings() -> None:
    guard = SQLGuard(max_rows=20)

    result = guard.validate(
        "SELECT doc_id FROM embeddings ORDER BY embedding <=> '[0,1]'::vector LIMIT 5"
    )

    assert result.allowed is True
    assert "embedding <=> CAST('[0,1]' AS VECTOR)" in result.safe_sql
