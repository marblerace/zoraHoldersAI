from __future__ import annotations

from types import SimpleNamespace

from app.config import Settings
from llm.anthropic_client import AnthropicClient
from llm.openai_client import OpenAIClient
from llm.types import Message, ToolDefinition

TOOL = ToolDefinition(
    name="run_sql",
    description="Run SQL",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
)


class Dumpable(SimpleNamespace):
    def model_dump(self, **_: object):
        return dict(self.__dict__)


class FakeResponses:
    def __init__(self, response) -> None:
        self.response = response
        self.requests: list[dict] = []

    def create(self, **request):
        self.requests.append(request)
        return self.response


class FakeMessages(FakeResponses):
    pass


def test_openai_adapter_uses_strict_responses_tools_and_normalizes_usage() -> None:
    output = Dumpable(
        type="function_call",
        call_id="call-1",
        name="run_sql",
        arguments='{"query":"SELECT 1"}',
    )
    response = SimpleNamespace(
        output=[output],
        output_text="",
        usage=SimpleNamespace(
            input_tokens=120,
            output_tokens=30,
            input_tokens_details=SimpleNamespace(cached_tokens=20),
        ),
    )
    responses = FakeResponses(response)
    sdk = SimpleNamespace(responses=responses)
    settings = Settings(_env_file=None, openai_model="gpt-5.6-terra")
    client = OpenAIClient(settings, api_key="test", sdk_client=sdk)

    completion = client.complete(
        [Message(role="system", content="system"), Message(role="user", content="question")],
        [TOOL],
    )

    request = responses.requests[0]
    assert request["tools"][0]["strict"] is True
    assert request["parallel_tool_calls"] is False
    assert request["max_tool_calls"] == 1
    assert request["store"] is False
    assert completion.tool_calls[0].arguments == {"query": "SELECT 1"}
    assert completion.usage.input_tokens == 100
    assert completion.usage.cache_read_input_tokens == 20
    assert completion.provider_payload[0]["call_id"] == "call-1"


def test_anthropic_adapter_uses_input_schema_and_normalizes_tool_use() -> None:
    block = Dumpable(
        type="tool_use",
        id="tool-1",
        name="run_sql",
        input={"query": "SELECT 1"},
    )
    response = SimpleNamespace(
        content=[block],
        usage=SimpleNamespace(
            input_tokens=80,
            output_tokens=20,
            cache_read_input_tokens=10,
            cache_creation_input_tokens=5,
        ),
    )
    messages = FakeMessages(response)
    sdk = SimpleNamespace(messages=messages)
    settings = Settings(_env_file=None, anthropic_model="claude-sonnet-5")
    client = AnthropicClient(settings, api_key="test", sdk_client=sdk)

    completion = client.complete(
        [Message(role="system", content="system"), Message(role="user", content="question")],
        [TOOL],
    )

    request = messages.requests[0]
    assert request["tools"][0]["input_schema"] == TOOL.parameters
    assert request["tool_choice"]["disable_parallel_tool_use"] is True
    assert completion.tool_calls[0].arguments == {"query": "SELECT 1"}
    assert completion.usage.input_tokens == 80
    assert completion.usage.cache_read_input_tokens == 10
    assert completion.usage.cache_write_input_tokens == 5
