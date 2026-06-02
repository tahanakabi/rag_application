"""Shared data structures for the FAQ RAG pipeline."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class Chunk:
    """A single retrievable unit derived from one FAQ question/answer pair."""

    id: str
    section: str          # the "## ..." category the Q/A belongs to
    question: str         # the "### ..." question text
    answer: str           # answer text for this (sub-)chunk
    ocr_text: str = ""    # text extracted from embedded base64 images via OCR
    sub_index: int = 0    # index of this sub-chunk within a long answer
    n_subchunks: int = 1  # total sub-chunks the answer was split into

    @property
    def text(self) -> str:
        """The full text used for embedding and shown to the LLM.

        The question is always prepended so each sub-chunk keeps its context.
        OCR text (tables/charts) is appended when present.
        """
        parts = [f"Question: {self.question}", f"Answer: {self.answer}"]
        if self.ocr_text.strip():
            parts.append(f"Image content (OCR): {self.ocr_text.strip()}")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


def save_chunks(chunks: Iterable[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(Chunk.from_dict(json.loads(line)))
    return chunks

