"""Embedding model wrapper producing L2-normalised float32 vectors."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..device import resolve_device


@lru_cache(maxsize=4)
def _load_model(model_name: str, trust_remote_code: bool, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_name, trust_remote_code=trust_remote_code, device=device
    )


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class Embedder:
    """Encode texts to L2-normalised float32 vectors."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        trust_remote_code: bool = False,
        query_prefix: str = "",
        doc_prefix: str = "",
        device: str = "auto",
    ):
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.device = resolve_device(device)
        # BGE models benefit from an instruction prefix on the query side.
        if not query_prefix and "bge" in model_name.lower():
            query_prefix = "Represent this sentence for searching relevant passages: "
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model(
                self.model_name, self.trust_remote_code, self.device
            )
        return self._model

    @property
    def full_dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def _encode(self, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
        inputs = [prefix + t for t in texts] if prefix else texts
        vecs = self.model.encode(
            inputs,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_documents(
        self, texts: list[str], batch_size: int = 32
    ) -> np.ndarray:
        vecs = self._encode(texts, self.doc_prefix, batch_size)
        return _l2_normalize(vecs).astype(np.float32)

    def encode_queries(
        self, texts: list[str], batch_size: int = 32
    ) -> np.ndarray:
        vecs = self._encode(texts, self.query_prefix, batch_size)
        return _l2_normalize(vecs).astype(np.float32)

