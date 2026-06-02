"""Turn parsed Q/A records into embedding-ready Chunks.

Chunking strategy:
  * one Q/A pair -> one chunk by default (keeps the natural question/answer split)
  * long answers are split into smaller sub-chunks on sentence/paragraph
    boundaries with a token budget and small overlap
  * the question is prepended to every sub-chunk (see Chunk.text) so context is
    never lost
  * embedded base64 images are OCR'd; the extracted text is attached to the
    sub-chunk where the image occurred and the heavy markup is stripped out
"""
from __future__ import annotations

import re
from pathlib import Path

from ..models import Chunk
from .image_ocr import OcrCache, ocr_text_for, strip_image_markup
from .parser import QARecord

# token estimation without a hard tiktoken dependency
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:  # noqa: BLE001 - tiktoken optional / offline
    def count_tokens(text: str) -> int:
        # ~4 chars per token heuristic
        return max(1, len(text) // 4)


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras or [text.strip()]


def _pack_sentences(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Greedily pack paragraphs/sentences into <= max_tokens windows."""
    units: list[str] = []
    for para in _split_paragraphs(text):
        if count_tokens(para) <= max_tokens:
            units.append(para)
        else:
            units.extend(s.strip() for s in _SENT_SPLIT.split(para) if s.strip())

    windows: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for unit in units:
        utok = count_tokens(unit)
        if cur and cur_tok + utok > max_tokens:
            windows.append(" ".join(cur))
            # build overlap tail
            tail: list[str] = []
            tail_tok = 0
            for u in reversed(cur):
                t = count_tokens(u)
                if tail_tok + t > overlap:
                    break
                tail.insert(0, u)
                tail_tok += t
            cur = tail[:]
            cur_tok = tail_tok
        cur.append(unit)
        cur_tok += utok
    if cur:
        windows.append(" ".join(cur))
    return windows or [text.strip()]


def build_chunks(
    records: list[QARecord],
    ocr_cache: OcrCache,
    ocr_engine: str = "tesseract",
    max_tokens: int = 384,
    overlap: int = 48,
) -> list[Chunk]:
    """Convert Q/A records into Chunks, splitting long answers and OCR'ing images."""
    chunks: list[Chunk] = []
    for qi, rec in enumerate(records):
        ocr = ocr_text_for(rec.answer, ocr_cache, engine=ocr_engine)
        clean_answer = strip_image_markup(rec.answer)

        windows = _pack_sentences(clean_answer, max_tokens=max_tokens, overlap=overlap)
        n = len(windows)
        for si, window in enumerate(windows):
            chunks.append(
                Chunk(
                    id=f"q{qi:04d}_s{si}",
                    section=rec.section,
                    question=rec.question,
                    answer=window,
                    # attach OCR only to the first sub-chunk to avoid duplication
                    ocr_text=ocr if si == 0 else "",
                    sub_index=si,
                    n_subchunks=n,
                )
            )
    return chunks


def ingest_markdown(
    md_path: Path,
    ocr_cache_path: Path,
    ocr_engine: str = "tesseract",
    max_tokens: int = 384,
    overlap: int = 48,
) -> list[Chunk]:
    """Full ingest: parse -> OCR -> chunk. Returns the chunk list."""
    from .parser import parse_markdown

    records = parse_markdown(md_path)
    cache = OcrCache(ocr_cache_path)
    chunks = build_chunks(
        records, cache, ocr_engine=ocr_engine, max_tokens=max_tokens, overlap=overlap
    )
    cache.flush()
    return chunks

