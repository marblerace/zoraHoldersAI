"""Fail-closed SQL validation using a Postgres-aware abstract syntax tree."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlglot import exp, parse
from sqlglot.errors import ParseError

from observability.tracing import start_span

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_TABLES = frozenset({"tokens", "holders", "transfers", "embeddings"})
FORBIDDEN_NODE_NAMES = frozenset(
    {
        "Alter",
        "Analyze",
        "Attach",
        "Cache",
        "Command",
        "Commit",
        "Copy",
        "Create",
        "Delete",
        "Detach",
        "Drop",
        "Execute",
        "Grant",
        "Insert",
        "Into",
        "LoadData",
        "Lock",
        "Merge",
        "Pragma",
        "Rollback",
        "Set",
        "Transaction",
        "TruncateTable",
        "Uncache",
        "Update",
        "Use",
    }
)
FORBIDDEN_FUNCTIONS = frozenset(
    {
        "current_setting",
        "dblink",
        "dblink_connect",
        "dblink_exec",
        "lo_export",
        "lo_import",
        "pg_ls_dir",
        "pg_read_binary_file",
        "pg_read_file",
        "pg_sleep",
        "currval",
        "lastval",
        "nextval",
        "setval",
        "set_config",
    }
)


@dataclass(frozen=True, slots=True)
class GuardResult:
    """Decision returned for every proposed query."""

    allowed: bool
    reason: str
    safe_sql: str


class SQLGuard:
    """Allow one bounded SELECT over the explicit analytics table allowlist."""

    def __init__(
        self,
        *,
        max_rows: int = 1000,
        allowed_tables: frozenset[str] = DEFAULT_ALLOWED_TABLES,
        max_query_characters: int = 20_000,
    ) -> None:
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        self._max_rows = max_rows
        self._allowed_tables = frozenset(table.lower() for table in allowed_tables)
        self._max_query_characters = max_query_characters

    def validate(self, query: str) -> GuardResult:
        """Parse, inspect, and cap a query, returning a structured decision."""

        query_length = len(query) if isinstance(query, str) else 0
        with start_span("guard.validate", query_length=query_length) as span:
            result = self._validate(query)
            span.set_attribute("guard_decision", "allowed" if result.allowed else "blocked")
            span.set_attribute("guard_reason", result.reason)
            return result

    def _validate(self, query: str) -> GuardResult:
        """Implement validation separately so tracing captures every early return."""

        if not isinstance(query, str) or not query.strip():
            return self._reject("Query is empty")
        if len(query) > self._max_query_characters:
            return self._reject("Query exceeds the maximum allowed length")

        try:
            statements = [statement for statement in parse(query, read="postgres") if statement]
        except (ParseError, ValueError) as exc:
            return self._reject(f"SQL parse error: {exc}")

        if len(statements) != 1:
            return self._reject("Exactly one SQL statement is required")

        statement = statements[0]
        if not isinstance(statement, exp.Query):
            return self._reject("Only SELECT queries are allowed")
        if not statement.find(exp.Select):
            return self._reject("Only SELECT queries are allowed")

        forbidden_node = self._find_forbidden_node(statement)
        if forbidden_node:
            return self._reject(f"Forbidden SQL operation: {forbidden_node}")

        table_reason = self._validate_tables(statement)
        if table_reason:
            return self._reject(table_reason)

        function_reason = self._validate_functions(statement)
        if function_reason:
            return self._reject(function_reason)

        limit_reason = self._apply_limit(statement)
        if limit_reason:
            return self._reject(limit_reason)

        safe_sql = statement.sql(dialect="postgres", pretty=False)
        vector_error, safe_sql = self._restore_pgvector_cosine_operator(
            query,
            statement,
            safe_sql,
        )
        if vector_error:
            return self._reject(vector_error)

        return GuardResult(
            allowed=True,
            reason="Query is a single bounded read-only SELECT",
            safe_sql=safe_sql,
        )

    @staticmethod
    def _find_forbidden_node(statement: exp.Expression) -> str | None:
        for node in statement.walk():
            node_name = type(node).__name__
            if node_name in FORBIDDEN_NODE_NAMES:
                return node_name.upper()
        return None

    def _validate_tables(self, statement: exp.Expression) -> str | None:
        cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
        for table in statement.find_all(exp.Table):
            table_name = table.name.lower()
            schema_name = table.db.lower() if table.db else ""
            catalog_name = table.catalog.lower() if table.catalog else ""

            if not schema_name and not catalog_name and table_name in cte_names:
                continue
            if catalog_name:
                return f"Cross-database reference is forbidden: {catalog_name}"
            if schema_name in {"information_schema", "pg_catalog"}:
                return f"System schema is forbidden: {schema_name}"
            if schema_name and schema_name != "public":
                return f"Schema is not allowlisted: {schema_name}"
            if table_name.startswith("pg_"):
                return f"System table is forbidden: {table_name}"
            if table_name not in self._allowed_tables:
                return f"Table is not allowlisted: {table_name}"
        return None

    @staticmethod
    def _validate_functions(statement: exp.Expression) -> str | None:
        for function in statement.find_all(exp.Func):
            if isinstance(function, exp.Anonymous):
                function_name = function.name.lower()
            else:
                function_name = function.sql_name().lower()
            if function_name in FORBIDDEN_FUNCTIONS or function_name.startswith(
                ("dblink_", "lo_", "pg_")
            ):
                return f"Function is forbidden: {function_name}"
        return None

    def _apply_limit(self, statement: exp.Query) -> str | None:
        limit = statement.args.get("limit")
        if limit is None:
            statement.limit(self._max_rows, copy=False)
            return None

        expression = limit.args.get("count") if isinstance(limit, exp.Fetch) else limit.expression
        if not isinstance(expression, exp.Literal) or not expression.is_int:
            return "LIMIT must be a non-negative integer literal"
        value = int(expression.this)
        if value < 0:
            return "LIMIT must be a non-negative integer literal"
        if value > self._max_rows:
            limit_key = "count" if isinstance(limit, exp.Fetch) else "expression"
            limit.set(limit_key, exp.Literal.number(self._max_rows))
        return None

    @staticmethod
    def _restore_pgvector_cosine_operator(
        original_query: str,
        statement: exp.Expression,
        safe_sql: str,
    ) -> tuple[str | None, str]:
        """Preserve pgvector ``<=>`` after SQLGlot's null-safe-equality parse.

        SQLGlot maps the shared token to ``IS NOT DISTINCT FROM`` in PostgreSQL.
        Restoration is fail-closed and limited to the allowlisted embeddings column.
        """

        raw_count = original_query.count("<=>")
        if raw_count == 0:
            return None, safe_sql
        nodes = list(statement.find_all(exp.NullSafeEQ))
        if len(nodes) != raw_count:
            return "Unsupported use of the pgvector cosine operator", safe_sql
        if not any(table.name.lower() == "embeddings" for table in statement.find_all(exp.Table)):
            return "pgvector cosine distance is only allowed on embeddings", safe_sql
        for node in nodes:
            columns = {column.name.lower() for column in node.find_all(exp.Column)}
            if "embedding" not in columns:
                return "pgvector cosine distance is only allowed on the embedding column", safe_sql
        marker = " IS NOT DISTINCT FROM "
        if safe_sql.count(marker) < raw_count:
            return "Could not safely serialize the pgvector cosine operator", safe_sql
        return None, safe_sql.replace(marker, " <=> ", raw_count)

    @staticmethod
    def _reject(reason: str) -> GuardResult:
        logger.warning("sql_guard_rejection reason=%s", reason)
        return GuardResult(allowed=False, reason=reason, safe_sql="")
