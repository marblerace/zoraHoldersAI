"""Anthropic Messages API adapter for the normalized LLM interface."""

from __future__ import annotations

from typing import Any

from anthropic import Anthropic

from app.config import Settings
from llm.client import LLMProviderError
from llm.types import Completion, Message, TokenUsage, ToolCall, ToolDefinition


class AnthropicClient:
    """Translate normalized messages to Anthropic content and tool-use blocks."""

    provider = "anthropic"

    def __init__(
        self,
        settings: Settings,
        *,
        api_key: str,
        sdk_client: Any | None = None,
    ) -> None:
        self.model = settings.anthropic_model
        self._max_output_tokens = settings.llm_max_output_tokens
        self._client = sdk_client or Anthropic(
            api_key=api_key,
            timeout=settings.llm_timeout_seconds,
        )

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> Completion:
        system, provider_messages = self._translate_messages(messages)
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_output_tokens,
            "system": system,
            "messages": provider_messages,
        }
        if tools:
            request["tools"] = [self._translate_tool(tool) for tool in tools]
            request["tool_choice"] = {
                "type": "auto",
                "disable_parallel_tool_use": True,
            }

        try:
            response = self._client.messages.create(**request)
        except Exception as exc:
            raise LLMProviderError(f"Anthropic request failed: {exc}") from exc

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                arguments = block.input if isinstance(block.input, dict) else {}
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=arguments))

        usage = response.usage
        normalized_usage = TokenUsage(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_read_input_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_write_input_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        )
        provider_payload = tuple(
            block.model_dump(mode="json", exclude_none=True) for block in response.content
        )
        return Completion(
            text="\n".join(part.strip() for part in text_parts if part.strip()),
            tool_calls=tuple(tool_calls),
            usage=normalized_usage,
            provider_payload=provider_payload,
        )

    @staticmethod
    def _translate_tool(tool: ToolDefinition) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }

    @staticmethod
    def _translate_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        system = "\n\n".join(message.content for message in messages if message.role == "system")
        provider_messages: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                continue
            if message.role == "assistant":
                if message.provider_payload:
                    content: Any = list(message.provider_payload)
                else:
                    content = []
                    if message.content:
                        content.append({"type": "text", "text": message.content})
                    content.extend(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                        for call in message.tool_calls
                    )
                provider_messages.append({"role": "assistant", "content": content})
            elif message.role == "tool":
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content,
                    "is_error": message.is_error,
                }
                if (
                    provider_messages
                    and provider_messages[-1]["role"] == "user"
                    and isinstance(provider_messages[-1]["content"], list)
                    and all(
                        block.get("type") == "tool_result"
                        for block in provider_messages[-1]["content"]
                    )
                ):
                    provider_messages[-1]["content"].append(tool_result)
                else:
                    provider_messages.append({"role": "user", "content": [tool_result]})
            else:
                provider_messages.append({"role": "user", "content": message.content})
        return system, provider_messages
