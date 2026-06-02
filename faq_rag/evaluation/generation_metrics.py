"""Generation quality metrics for the RAG answers.

Primary path uses ragas (faithfulness, answer relevancy, context precision).
If ragas is unavailable or no judge LLM is configured, we fall back to
lightweight lexical proxies so the pipeline always produces numbers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class GenSample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _lexical_faithfulness(answer: str, contexts: list[str]) -> float:
    """Fraction of answer tokens supported by the retrieved context."""
    a = _tokens(answer)
    if not a:
        return 0.0
    ctx = set().union(*[_tokens(c) for c in contexts]) if contexts else set()
    return len(a & ctx) / len(a)


def _lexical_relevancy(answer: str, question: str) -> float:
    q = _tokens(question)
    a = _tokens(answer)
    if not q or not a:
        return 0.0
    return len(q & a) / len(q)


def fallback_metrics(samples: list[GenSample]) -> dict[str, float]:
    if not samples:
        return {"faithfulness_lex": 0.0, "answer_relevancy_lex": 0.0}
    faith = sum(_lexical_faithfulness(s.answer, s.contexts) for s in samples) / len(samples)
    rel = sum(_lexical_relevancy(s.answer, s.question) for s in samples) / len(samples)
    return {"faithfulness_lex": faith, "answer_relevancy_lex": rel}


def ragas_metrics(samples: list[GenSample], openai_api_key: str = "") -> dict[str, float] | None:
    """Compute ragas metrics; return None if ragas / judge LLM not available."""
    if not openai_api_key:
        return None
    try:
        import os

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        os.environ.setdefault("OPENAI_API_KEY", openai_api_key)
        ds = Dataset.from_list(
            [
                {
                    "question": s.question,
                    "answer": s.answer,
                    "contexts": s.contexts,
                    "ground_truth": s.ground_truth or s.answer,
                }
                for s in samples
            ]
        )
        result = evaluate(
            ds, metrics=[faithfulness, answer_relevancy, context_precision]
        )
        return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
    except Exception:  # noqa: BLE001 - ragas optional / network / version drift
        return None


def evaluate_generation(samples: list[GenSample], openai_api_key: str = "") -> dict[str, float]:
    metrics = ragas_metrics(samples, openai_api_key=openai_api_key)
    if metrics:
        return metrics
    return fallback_metrics(samples)

