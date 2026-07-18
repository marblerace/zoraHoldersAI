"""Bounded text-to-SQL agent loop."""

from __future__ import annotations

from typing import Any

__all__ = ["AgentResult", "TextToSQLAgent"]


def __getattr__(name: str) -> Any:
    """Lazily expose the service without importing it when a submodule is requested."""

    if name in __all__:
        from agent.service import AgentResult, TextToSQLAgent

        return {"AgentResult": AgentResult, "TextToSQLAgent": TextToSQLAgent}[name]
    raise AttributeError(name)
