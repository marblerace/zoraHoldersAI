"""OpenAI Responses API adapter for the normalized LLM interface."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app.config import Settings
from llm.client import LLMProviderError
from llm.types import Completion, Message, TokenUsage, ToolCall, ToolDefinition


class OpenAIClient:
    """Use strict function tools while preserving Responses API output items."""

    provider = "openai"

    def __init__(
        self,
        settings: Settings,
        *,
        api_key: str,
        sdk_client: Any | None = None,
    ) -> None:
        self.model = settings.openai_model
        self._max_output_tokens = settings.llm_max_output_tokens
        self._reasoning_effort = settings.openai_reasoning_effort
        self._client = sdk_client or OpenAI(
            api_key=api_key,
            timeout=settings.llm_timeout_seconds,
        )

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> Completion:
        instructions, input_items = self._translate_messages(messages)
        request: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "instructions": instructions,
            "max_output_tokens": self._max_output_tokens,
            "store": False,
        }
        if tools:
            request.update(
                tools=[self._translate_tool(tool) for tool in tools],
                tool_choice="auto",
                parallel_tool_calls=False,
                max_tool_calls=1,
            )
        if self._reasoning_effort != "none" and self.model.startswith("gpt-5"):
            request["reasoning"] = {"effort": self._reasoning_effort}

        try:
            response = self._client.responses.create(**request)
        except Exception as exc:
            raise LLMProviderError(f"OpenAI request failed: {exc}") from exc

        provider_payload = tuple(
            item.model_dump(mode="json", exclude_none=True) for item in response.output
        )
        tool_calls: list[ToolCall] = []
        for item in response.output:
            if getattr(item, "type", None) != "function_call":
                continue
            raw_arguments = getattr(item, "arguments", "{}")
            try:
                arguments = json.loads(raw_arguments)
            except (TypeError, json.JSONDecodeError):
                arguments = {"__invalid_json__": str(raw_arguments)}
            if not isinstance(arguments, dict):
                arguments = {"__invalid_json__": str(raw_arguments)}
            tool_calls.append(
                ToolCall(
                    id=item.call_id,
                    name=item.name,
                    arguments=arguments,
                )
            )

        usage = response.usage
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        input_details = getattr(usage, "input_tokens_details", None)
        cached_tokens = int(getattr(input_details, "cached_tokens", 0) or 0)
        normalized_usage = TokenUsage(
            input_tokens=max(input_tokens - cached_tokens, 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_read_input_tokens=cached_tokens,
        )
        return Completion(
            text=(response.output_text or "").strip(),
            tool_calls=tuple(tool_calls),
            usage=normalized_usage,
            provider_payload=provider_payload,
        )

    @staticmethod
    def _translate_tool(tool: ToolDefinition) -> dict[str, Any]:
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": True,
        }

    @staticmethod
    def _translate_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        instructions = "\n\n".join(
            message.content for message in messages if message.role == "system"
        )
        input_items: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                continue
            if message.role in {"user", "assistant"}:
                if message.role == "assistant" and message.provider_payload:
                    input_items.extend(message.provider_payload)
                    continue
                if message.content:
                    input_items.append({"role": message.role, "content": message.content})
                if message.role == "assistant":
                    input_items.extend(
                        {
                            "type": "function_call",
                            "call_id": call.id,
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        }
                        for call in message.tool_calls
                    )
                continue
            if message.role == "tool":
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": message.content,
                    }
                )
        return instructions, input_items
