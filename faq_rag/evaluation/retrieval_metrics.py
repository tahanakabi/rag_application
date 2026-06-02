"""Retrieval quality metrics: Hit@k, MRR, Recall@k, nDCG@k."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .eval_set import EvalExample


@dataclass
class RetrievalMetrics:
    hit_at_k: float
    mrr: float
    recall_at_k: float
    ndcg_at_k: float
    k: int
    n_queries: int

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}hit_at_{self.k}": self.hit_at_k,
            f"{prefix}mrr": self.mrr,
            f"{prefix}recall_at_{self.k}": self.recall_at_k,
            f"{prefix}ndcg_at_{self.k}": self.ndcg_at_k,
        }


def _dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def evaluate_retriever(retriever, examples: list[EvalExample], k: int = 5) -> RetrievalMetrics:
    hits = 0
    rr_sum = 0.0
    recall_sum = 0.0
    ndcg_sum = 0.0

    for ex in examples:
        relevant = set(ex.relevant_ids)
        results = retriever.retrieve(ex.query, top_k=k)
        retrieved_ids = [r.chunk.id for r in results]

        gains = [1 if cid in relevant else 0 for cid in retrieved_ids]

        if any(gains):
            hits += 1
            first = gains.index(1)
            rr_sum += 1.0 / (first + 1)

        if relevant:
            recall_sum += sum(gains) / min(len(relevant), k)

        ideal = sorted(gains, reverse=True)
        idcg = _dcg(ideal)
        ndcg_sum += (_dcg(gains) / idcg) if idcg > 0 else 0.0

    n = max(1, len(examples))
    return RetrievalMetrics(
        hit_at_k=hits / n,
        mrr=rr_sum / n,
        recall_at_k=recall_sum / n,
        ndcg_at_k=ndcg_sum / n,
        k=k,
        n_queries=len(examples),
    )

