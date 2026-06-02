"""Build a question -> relevant-chunk gold set for retrieval evaluation.

Two complementary strategies:
  * auto-seed: each question is its own query; the relevant chunks are all
    sub-chunks derived from that question (same ``q####`` prefix).
  * paraphrase: lightly perturb the question (drop the leading wh-word / make it
    keyword-like) to create harder queries that still map to the same chunks.
A held-out JSON file can also be provided to add manually-curated pairs.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..models import Chunk


@dataclass
class EvalExample:
    query: str
    relevant_ids: list[str] = field(default_factory=list)


def _base_id(chunk_id: str) -> str:
    # "q0007_s2" -> "q0007"
    return chunk_id.split("_s")[0]


def _keywordize(question: str) -> str:
    q = re.sub(r"^(what|how|when|where|which|why|who|can|should|is|are|does|do)\b",
               "", question.strip(), flags=re.IGNORECASE).strip(" ?")
    return q or question


def build_eval_set(
    chunks: list[Chunk],
    n: int = 40,
    include_paraphrase: bool = True,
    seed: int = 13,
) -> list[EvalExample]:
    rng = random.Random(seed)

    # group chunk ids by their originating question
    groups: dict[str, list[str]] = {}
    question_of: dict[str, str] = {}
    for c in chunks:
        b = _base_id(c.id)
        groups.setdefault(b, []).append(c.id)
        question_of[b] = c.question

    bases = list(groups.keys())
    rng.shuffle(bases)
    bases = bases[: min(n, len(bases))]

    examples: list[EvalExample] = []
    for b in bases:
        examples.append(EvalExample(query=question_of[b], relevant_ids=groups[b]))
        if include_paraphrase:
            kw = _keywordize(question_of[b])
            if kw and kw.lower() != question_of[b].lower():
                examples.append(EvalExample(query=kw, relevant_ids=groups[b]))
    return examples


def save_eval_set(examples: list[EvalExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([e.__dict__ for e in examples], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_eval_set(path: Path) -> list[EvalExample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EvalExample(**d) for d in data]

