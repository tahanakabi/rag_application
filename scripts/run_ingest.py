"""Ingest the FAQ markdown: parse -> OCR -> chunk -> embed -> FAISS.

Usage:
    python -m scripts.run_ingest
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faq_rag.config import FAQ_DOC, get_settings
from faq_rag.ingestion.chunker import ingest_markdown
from faq_rag.models import save_chunks
from faq_rag.pipeline import build_embedder, build_faiss, embed_chunks


def main() -> None:
    settings = get_settings()
    print(f"Parsing & chunking: {FAQ_DOC}")
    ocr_cache_path = settings.data_path / "ocr_cache.json"
    chunks = ingest_markdown(
        FAQ_DOC,
        ocr_cache_path=ocr_cache_path,
        ocr_engine=settings.ocr_engine,
    )
    print(f"  -> {len(chunks)} chunks")
    save_chunks(chunks, settings.chunks_file)
    print(f"  -> saved chunks to {settings.chunks_file}")

    print(f"Embedding with {settings.embed_model} ...")
    embedder = build_embedder(settings)
    vectors = embed_chunks(embedder, chunks)
    print(f"  -> vectors {vectors.shape}")

    # The FAISS index doubles as the canonical store of the raw document
    # embeddings; downstream eval/app reconstruct vectors from it (no extra file).
    faiss_store = build_faiss(chunks, vectors)
    faiss_store.save(settings.faiss_index)
    print(f"  -> FAISS index saved to {settings.faiss_index}")


    print("Ingestion complete.")


if __name__ == "__main__":
    main()

