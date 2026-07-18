"""Structured request observability."""

from __future__ import annotations

from typing import Any

__all__ = ["QueryLogRecord", "record_query"]


def __getattr__(name: str) -> Any:
    """Keep public conveniences without eagerly coupling tracing to database logging."""

    if name in __all__:
        from observability.query_log import QueryLogRecord, record_query

        return {"QueryLogRecord": QueryLogRecord, "record_query": record_query}[name]
    raise AttributeError(name)
