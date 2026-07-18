from __future__ import annotations

import json
import subprocess

import pytest

from app.config import Settings
from llm.claude_code_client import ClaudeCodeClient
from llm.client import LLMConfigurationError, LLMProviderError, create_llm_client
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


class FakeRunner:
    def __init__(
        self,
        payload: dict,
        *,
        returncode: int = 0,
        stderr: str = "",
    ) -> None:
        self.payload = payload
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[tuple[list[str], dict[str, str], float]] = []

    def __call__(
        self,
        command: list[str],
        env: dict[str, str],
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((command, env, timeout))
        return subprocess.CompletedProcess(
            command,
            self.returncode,
            stdout=json.dumps(self.payload),
            stderr=self.stderr,
        )


def test_claude_code_adapter_normalizes_tool_call_and_subscription_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-cli")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "must-not-reach-cli")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    runner = FakeRunner(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "session_id": "session-1",
            "structured_output": {
                "action": "run_sql",
                "text": "",
                "arguments": {"query": "SELECT 1"},
            },
            "usage": {
                "input_tokens": 80,
                "output_tokens": 20,
                "cache_read_input_tokens": 10,
                "cache_creation_input_tokens": 5,
            },
        }
    )
    settings = Settings(
        _env_file=None,
        llm_provider="claude_code",
        claude_code_model="sonnet",
    )
    client = ClaudeCodeClient(settings, runner=runner)

    completion = client.complete(
        [Message(role="system", content="system"), Message(role="user", content="question")],
        [TOOL],
    )

    command, env, timeout = runner.calls[0]
    assert command[0] == "claude"
    assert command[1] == "-p"
    assert "--safe-mode" in command
    assert "--no-session-persistence" in command
    assert command[command.index("--tools") + 1] == ""
    output_schema = json.loads(command[command.index("--json-schema") + 1])
    assert output_schema["properties"]["action"]["enum"] == ["answer", "run_sql"]
    assert output_schema["properties"]["arguments"] == TOOL.parameters
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert "CLAUDE_CODE_USE_BEDROCK" not in env
    assert timeout == 180
    assert completion.text == ""
    assert completion.tool_calls[0].id == "claude-code-session-1"
    assert completion.tool_calls[0].name == "run_sql"
    assert completion.tool_calls[0].arguments == {"query": "SELECT 1"}
    assert completion.usage.input_tokens == 80
    assert completion.usage.output_tokens == 20
    assert completion.usage.cache_read_input_tokens == 10
    assert completion.usage.cache_write_input_tokens == 5


def test_claude_code_adapter_normalizes_final_answer_from_result_fallback() -> None:
    runner = FakeRunner(
        {
            "type": "result",
            "subtype": "success",
            "result": json.dumps({"answer": "There are 12 holders."}),
        }
    )
    settings = Settings(_env_file=None, llm_provider="claude_code")
    client = ClaudeCodeClient(settings, runner=runner)

    completion = client.complete([Message(role="user", content="summarize")], [])

    assert completion.text == "There are 12 holders."
    assert completion.tool_calls == ()


def test_claude_code_adapter_surfaces_login_error() -> None:
    runner = FakeRunner({}, returncode=1, stderr="Error: Not logged in")
    settings = Settings(_env_file=None, llm_provider="claude_code")
    client = ClaudeCodeClient(settings, runner=runner)

    with pytest.raises(LLMProviderError, match=r"claude auth login"):
        client.complete([Message(role="user", content="question")], [])


def test_factory_builds_claude_code_without_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("llm.client.shutil.which", lambda _: "/usr/local/bin/claude")
    settings = Settings(
        _env_file=None,
        llm_provider="claude_code",
        anthropic_api_key=None,
    )

    client = create_llm_client(settings)

    assert isinstance(client, ClaudeCodeClient)
    assert client.provider == "claude_code"
    assert client.model == "sonnet"


def test_factory_explains_when_claude_cli_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("llm.client.shutil.which", lambda _: None)
    settings = Settings(_env_file=None, llm_provider="claude_code")

    with pytest.raises(LLMConfigurationError, match="run the API on the host"):
        create_llm_client(settings)
