"""Prompt templates that constrain answers strictly to the FAQ context."""
from __future__ import annotations

from ..retrieval.dense import RetrievalResult

SYSTEM_PROMPT = (
    "You are a helpful FAQ assistant. Answer the user's question using ONLY the "
    "information in the provided context passages. "
    "If the answer is not contained in the context, reply exactly: "
    "\"I don't have information about that in the FAQ.\" "
    "Do not use outside knowledge. Be concise and factual. "
    "When the context includes table or chart text (from images), you may use it."
)

REFUSAL = "I don't have information about that in the FAQ."


def format_context(results: list[RetrievalResult]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[Passage {i} | section: {r.chunk.section}]\n{r.chunk.text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, results: list[RetrievalResult]) -> str:
    context = format_context(results)
    return (
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )

