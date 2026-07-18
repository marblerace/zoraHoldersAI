from __future__ import annotations

import json

from eval.judge import GroundednessJudge
from llm.types import Completion, Message, TokenUsage, ToolDefinition


class FakeJudgeLLM:
    provider = "test"
    model = "test"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[list[Message], list[ToolDefinition]]] = []

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> Completion:
        self.calls.append((messages, tools))
        return Completion(
            text=self.response,
            tool_calls=(),
            usage=TokenUsage(input_tokens=10, output_tokens=2),
        )


def test_judge_prompt_separates_faithfulness_from_correctness() -> None:
    llm = FakeJudgeLLM('{"grounded":1}')
    judge = GroundednessJudge(llm)

    grounded, usage = judge.score(
        question="How many holders acquired their position recently?",
        answer="The query returned 12.",
        rows=({"count": 12},),
    )

    assert grounded == 1
    assert usage.output_tokens == 2
    messages, tools = llm.calls[0]
    assert tools == []
    assert "Judge faithfulness only" in messages[0].content
    assert "Do not decide whether clarification was necessary" in messages[0].content
    assert json.loads(messages[1].content)["rows"] == [{"count": 12}]


def test_judge_rejects_invalid_or_out_of_range_responses() -> None:
    for response in ("not json", '{"grounded":2}', "{}"):
        grounded, _ = GroundednessJudge(FakeJudgeLLM(response)).score(
            question="question",
            answer="answer",
            rows=(),
        )
        assert grounded is None
