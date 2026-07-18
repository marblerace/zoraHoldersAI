"""Optional LLM judge for natural-language answer groundedness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm.types import LLMClient, Message, TokenUsage


class GroundednessJudge:
    """Score whether an answer makes only claims supported by its tool rows."""

    def __init__(self, llm: LLMClient, prompt_path: Path | None = None) -> None:
        self._llm = llm
        path = prompt_path or Path(__file__).with_name("judge_prompt.txt")
        self._prompt = path.read_text(encoding="utf-8")

    def score(
        self,
        *,
        question: str,
        answer: str,
        rows: tuple[dict[str, Any], ...],
    ) -> tuple[int | None, TokenUsage]:
        payload = json.dumps(
            {"question": question, "answer": answer, "rows": rows},
            ensure_ascii=False,
            default=str,
        )
        completion = self._llm.complete(
            [Message(role="system", content=self._prompt), Message(role="user", content=payload)],
            [],
        )
        try:
            parsed = json.loads(completion.text)
            grounded = int(parsed["grounded"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None, completion.usage
        return (grounded if grounded in {0, 1} else None), completion.usage
