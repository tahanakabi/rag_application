"""LLM backends with a common ``generate(system, user)`` interface."""
from __future__ import annotations

from typing import Protocol


class LLM(Protocol):
    name: str

    def generate(self, system: str, user: str) -> str:
        ...


class OpenAILLM:
    """OpenAI chat completion backend; key read from the environment."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = "", temperature: float = 0.0):
        from openai import OpenAI

        self.name = f"openai:{model}"
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key or None)  # falls back to OPENAI_API_KEY env

    def generate(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

