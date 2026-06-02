"""In-memory FAISS index over normalised embeddings (cosine via inner product)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class FaissStore:
    def __init__(self, dim: int):
        import faiss

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product == cosine on unit vectors
        self.ids: list[str] = []

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        assert vectors.shape[1] == self.dim, "embedding dim mismatch"
        self.index.add(vectors.astype(np.float32))
        self.ids.extend(ids)

    def get_all_vectors(self) -> np.ndarray:
        """Reconstruct all stored vectors as an (n, dim) matrix (flat index only).

        Lets the FAISS index act as the single source of truth for the raw
        document embeddings without a separate on-disk vector cache.
        """
        n = self.index.ntotal
        if n == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.index.reconstruct_n(0, n).astype(np.float32)

    def search(self, query: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        if query.ndim == 1:
            query = query.reshape(1, -1)
        scores, idxs = self.index.search(query.astype(np.float32), top_k)
        out: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            out.append((self.ids[idx], float(score)))
        return out

    def save(self, path: Path) -> None:
        import faiss

        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        with path.with_suffix(".ids.pkl").open("wb") as fh:
            pickle.dump({"ids": self.ids, "dim": self.dim}, fh)

    @classmethod
    def load(cls, path: Path) -> "FaissStore":
        import faiss

        with path.with_suffix(".ids.pkl").open("rb") as fh:
            meta = pickle.load(fh)
        store = cls.__new__(cls)
        store.dim = meta["dim"]
        store.ids = meta["ids"]
        store.index = faiss.read_index(str(path))
        return store

