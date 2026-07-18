"""Server-side schema introspection for the model system prompt."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.config import Settings, get_settings
from db.core import connect_reader
from observability.tracing import start_span

ANALYTICS_TABLES = ("tokens", "holders", "transfers")


@dataclass(frozen=True, slots=True)
class SchemaSnapshot:
    schema_text: str
    last_synced_at: datetime | None


def introspect_schema(settings: Settings | None = None) -> SchemaSnapshot:
    """Build a compact schema description from information_schema, not source constants."""

    resolved = settings or get_settings()
    with start_span("db.execute", operation="schema_introspection") as span:
        with connect_reader(resolved) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT table_name, column_name, data_type, udt_name, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = ANY(%s)
                ORDER BY table_name, ordinal_position
                """,
                (list(ANALYTICS_TABLES),),
            )
            columns = cursor.fetchall()
            cursor.execute("SELECT MAX(last_synced_at) AS last_synced_at FROM tokens")
            freshness = cursor.fetchone()
        span.set_attribute("rows_returned", len(columns))

    by_table: dict[str, list[str]] = {table: [] for table in ANALYTICS_TABLES}
    for column in columns:
        data_type = column["data_type"]
        if data_type == "USER-DEFINED":
            data_type = column["udt_name"]
        nullable = "" if column["is_nullable"] == "YES" else " NOT NULL"
        by_table[column["table_name"]].append(f"  {column['column_name']} {data_type}{nullable}")

    missing = [table for table, definitions in by_table.items() if not definitions]
    if missing:
        raise RuntimeError(f"Analytics schema is missing required tables: {', '.join(missing)}")

    schema_text = "\n\n".join(
        f"CREATE TABLE {table} (\n" + ",\n".join(by_table[table]) + "\n);"
        for table in ANALYTICS_TABLES
    )
    return SchemaSnapshot(
        schema_text=schema_text,
        last_synced_at=freshness["last_synced_at"] if freshness else None,
    )
