"""Common types and the dense (semantic) retriever baseline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..ingestion.embeddings import Embedder
from ..models import Chunk


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        ...


class DenseRetriever:
    """Pure semantic search against a FAISS vector store."""

    name = "dense"

    def __init__(self, store, embedder: Embedder, chunks: list[Chunk]):
        self.store = store
        self.embedder = embedder
        self.by_id = {c.id: c for c in chunks}

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        qvec = self.embedder.encode_queries([query])[0]
        hits = self.store.search(qvec, top_k=top_k)
        return [RetrievalResult(self.by_id[cid], score) for cid, score in hits if cid in self.by_id]

