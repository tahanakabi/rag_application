"""High-level helpers to build vector stores and retrievers from chunks."""
from __future__ import annotations

import numpy as np

from .config import Settings
from .ingestion.embeddings import Embedder
from .models import Chunk
from .stores.faiss_store import FaissStore


def build_embedder(settings: Settings) -> Embedder:
    return Embedder(
        model_name=settings.embed_model,
        trust_remote_code=settings.embed_trust_remote_code,
        device=settings.device,
    )


def embed_chunks(embedder: Embedder, chunks: list[Chunk]) -> np.ndarray:
    return embedder.encode_documents([c.text for c in chunks])


def build_faiss(chunks: list[Chunk], vectors: np.ndarray) -> FaissStore:
    store = FaissStore(dim=vectors.shape[1])
    store.add([c.id for c in chunks], vectors)
    return store


def build_store(
    settings: Settings,
    chunks: list[Chunk],
    vectors: np.ndarray,
    collection: str = "faq_chunks",
) -> FaissStore:
    """Return a FAISS vector store for the given chunks and vectors."""
    return build_faiss(chunks, vectors)


def load_store_and_vectors(
    settings: Settings,
    chunks: list[Chunk],
    embedder: Embedder,
):
    """Return a ready search store and the full-dim document vectors.

    The persisted FAISS index (written at ingestion) is the single source of
    truth for the raw embeddings: we load it and reconstruct the vectors instead
    of re-embedding or keeping a separate vector file. If the index is missing or
    stale, we fall back to embedding the chunks once.
    """
    faiss_path = settings.faiss_index
    if faiss_path.exists() and faiss_path.with_suffix(".ids.pkl").exists():
        faiss_store = FaissStore.load(faiss_path)
        if len(faiss_store.ids) == len(chunks):
            vectors = faiss_store.get_all_vectors()
            return faiss_store, vectors
        print("[warn] persisted FAISS index stale (count mismatch); recomputing")

    vectors = embed_chunks(embedder, chunks)
    return build_faiss(chunks, vectors), vectors


def build_all_retrievers(
    settings: Settings,
    chunks: list[Chunk],
    embedder: Embedder,
    store,
    full_vectors: np.ndarray | None = None,
):
    """Construct the retrieval strategies sharing one full-dim vector store.

    ``full_vectors`` is accepted for call-site compatibility but is not required.
    """
    from .retrieval.dense import DenseRetriever
    from .retrieval.hybrid import HybridRetriever
    from .retrieval.rerank import RerankRetriever

    dense = DenseRetriever(store, embedder, chunks)
    hybrid = HybridRetriever(store, embedder, chunks)
    rerank = RerankRetriever(
        dense, model_name=settings.rerank_model, device=settings.device
    )

    return {
        "dense": dense,
        "hybrid_rrf": hybrid,
        "rerank": rerank,
    }

