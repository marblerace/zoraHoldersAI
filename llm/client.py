"""Validated factory for the configured model provider."""

from __future__ import annotations

import shutil

from app.config import Settings, get_settings
from llm.types import LLMClient


class LLMConfigurationError(RuntimeError):
    """Raised when the selected provider is not usable in this environment."""


class LLMProviderError(RuntimeError):
    """Raised when a provider request or response cannot be completed safely."""


def create_llm_client(settings: Settings | None = None) -> LLMClient:
    """Build the selected provider adapter without requiring keys at API startup."""

    resolved = settings or get_settings()
    if resolved.llm_provider == "anthropic":
        from llm.anthropic_client import AnthropicClient

        key = _secret_value(resolved.anthropic_api_key)
        if not key:
            raise LLMConfigurationError(
                "LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY for POST /ask"
            )
        return AnthropicClient(resolved, api_key=key)

    if resolved.llm_provider == "openai":
        from llm.openai_client import OpenAIClient

        key = _secret_value(resolved.openai_api_key)
        if not key:
            raise LLMConfigurationError("LLM_PROVIDER=openai requires OPENAI_API_KEY for POST /ask")
        return OpenAIClient(resolved, api_key=key)

    from llm.claude_code_client import ClaudeCodeClient

    if shutil.which(resolved.claude_code_command) is None:
        raise LLMConfigurationError(
            "LLM_PROVIDER=claude_code requires the local Claude Code CLI. "
            "Install it and run `claude auth login`, then run the API on the host "
            "rather than inside Docker."
        )
    return ClaudeCodeClient(resolved)


def _secret_value(secret: object) -> str:
    if secret is None:
        return ""
    getter = getattr(secret, "get_secret_value", None)
    return getter().strip() if getter else str(secret).strip()
