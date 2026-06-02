"""The RAG chain: retrieve -> build prompt -> generate, constrained to FAQ context."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..retrieval.dense import RetrievalResult
from .prompts import REFUSAL, SYSTEM_PROMPT, build_user_prompt


@dataclass
class RagResponse:
    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    results: list[RetrievalResult] = field(default_factory=list)


class RagChain:
    def __init__(self, retriever, llm, top_k: int = 4, min_score: float | None = None):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        # Optional score floor: if the best hit is below this, refuse early.
        self.min_score = min_score

    def retrieve(self, question: str) -> list[RetrievalResult]:
        """Run only the retrieval stage (no LLM call)."""
        return self.retriever.retrieve(question, top_k=self.top_k)

    def generate(self, question: str, results: list[RetrievalResult]) -> RagResponse:
        """Generate an answer from already-retrieved results."""
        if not results or (self.min_score is not None and results[0].score < self.min_score):
            return RagResponse(question=question, answer=REFUSAL, contexts=[], results=results)

        user_prompt = build_user_prompt(question, results)
        answer = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        contexts = [r.chunk.text for r in results]
        return RagResponse(question=question, answer=answer, contexts=contexts, results=results)

    def answer(self, question: str) -> RagResponse:
        results = self.retrieve(question)
        return self.generate(question, results)

