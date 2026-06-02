"""Hybrid retrieval: dense semantic + BM25 sparse, fused with Reciprocal Rank Fusion."""
from __future__ import annotations

import re

from ..ingestion.embeddings import Embedder
from ..models import Chunk
from .dense import RetrievalResult


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    """Combine dense vector search with BM25 lexical search via RRF."""

    name = "hybrid_rrf"

    def __init__(
        self,
        store,
        embedder: Embedder,
        chunks: list[Chunk],
        rrf_k: int = 60,
        candidate_k: int = 30,
    ):
        from rank_bm25 import BM25Okapi

        self.store = store
        self.embedder = embedder
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self._corpus_ids = [c.id for c in chunks]
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    def _dense_ranking(self, query: str) -> list[str]:
        qvec = self.embedder.encode_queries([query])[0]
        hits = self.store.search(qvec, top_k=self.candidate_k)
        return [cid for cid, _ in hits]

    def _sparse_ranking(self, query: str) -> list[str]:
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._corpus_ids[i] for i in order[: self.candidate_k]]

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        dense_ids = self._dense_ranking(query)
        sparse_ids = self._sparse_ranking(query)

        fused: dict[str, float] = {}
        for ranking in (dense_ids, sparse_ids):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [RetrievalResult(self.by_id[cid], score) for cid, score in ordered if cid in self.by_id]

