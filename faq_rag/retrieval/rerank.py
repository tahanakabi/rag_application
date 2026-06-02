"""Reranking retriever: fetch candidates then reorder with a cross-encoder."""
from __future__ import annotations

from functools import lru_cache

from ..device import resolve_device
from .dense import RetrievalResult


@lru_cache(maxsize=2)
def _load_reranker(model_name: str, device: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=device)


class RerankRetriever:
    """Wrap a base retriever and rerank its candidates with a cross-encoder."""

    name = "rerank"

    def __init__(self, base_retriever, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", candidate_k: int = 12, batch_size: int = 32, device: str = "auto"):
        self.base = base_retriever
        self.model_name = model_name
        self.candidate_k = candidate_k
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = _load_reranker(self.model_name, self.device)
        return self._reranker

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        candidates = self.base.retrieve(query, top_k=self.candidate_k)
        if not candidates:
            return []
        pairs = [(query, r.chunk.text) for r in candidates]
        scores = self.reranker.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        reranked = sorted(
            (RetrievalResult(c.chunk, float(s)) for c, s in zip(candidates, scores)),
            key=lambda r: r.score,
            reverse=True,
        )
        return reranked[:top_k]

