"""Interactive FAQ chatbot entrypoint.

CLI mode (default):
    python -m faq_rag.app --backend openai --retriever rerank

Streamlit mode:
    streamlit run faq_rag/app.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Quiet the noisy HF/transformers download + weight-loading output. Must be set
# before transformers/huggingface_hub get imported (they are imported lazily by
# the model loaders below).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# allow running both as `python -m faq_rag.app` and `streamlit run faq_rag/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faq_rag.config import get_settings
from faq_rag.models import load_chunks
from faq_rag.pipeline import build_all_retrievers, build_embedder, load_store_and_vectors
from faq_rag.rag.chain import RagChain


def build_chain(
    backend: str = "openai",
    retriever_name: str = "rerank",
    top_k: int = 4,
) -> RagChain:
    settings = get_settings()
    chunks = load_chunks(settings.chunks_file)
    embedder = build_embedder(settings)
    store, vectors = load_store_and_vectors(settings, chunks, embedder)
    retrievers = build_all_retrievers(
        settings, chunks, embedder, store, full_vectors=vectors
    )
    retriever = retrievers[retriever_name]

    if backend == "openai":
        from faq_rag.rag.llm_openai import OpenAILLM

        llm = OpenAILLM(model=settings.openai_model, api_key=settings.openai_api_key)
    else:
        from faq_rag.rag.llm_local import LocalLLM

        llm = LocalLLM(repo_id=settings.local_llm_repo, filename=settings.local_llm_file)

    chain = RagChain(retriever, llm, top_k=top_k)
    _warm_up(chain)
    return chain


def _warm_up(chain: RagChain) -> None:
    """Force model weights to load once now, instead of on the first question.

    The embedder and cross-encoder are created lazily; touching them here moves
    the one-time 'Loading weights' step to startup so it doesn't reappear
    mid-conversation.
    """
    embedder = getattr(chain.retriever, "embedder", None)
    if embedder is not None:
        _ = embedder.model
    base = getattr(chain.retriever, "base", None)
    reranker_owner = chain.retriever if hasattr(chain.retriever, "reranker") else base
    if reranker_owner is not None and hasattr(reranker_owner, "reranker"):
        _ = reranker_owner.reranker


def _print_retrieved(results) -> None:
    if not results:
        print("(no documents retrieved)\n")
        return
    print(f"Retrieved {len(results)} document(s):")
    for i, r in enumerate(results, 1):
        snippet = " ".join(r.chunk.text.split())
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        print(f"  [{i}] (section: {r.chunk.section}, score: {r.score:.3f})")
        print(f"      {snippet}")
    print()


def run_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["openai", "local"], default="local")
    parser.add_argument("--retriever", default="rerank")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    chain = build_chain(args.backend, args.retriever, args.top_k)
    print(f"FAQ chatbot ready ({args.backend} / {args.retriever}). Type 'exit' to quit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        # Retrieve and show the supporting documents before calling the LLM.
        results = chain.retrieve(q)
        _print_retrieved(results)
        resp = chain.generate(q, results)
        print(f"Bot: {resp.answer}\n")


def run_streamlit() -> None:
    import streamlit as st

    st.set_page_config(page_title="FAQ Chatbot", page_icon="💬")
    st.title("FAQ Chatbot")

    backend = st.sidebar.selectbox("LLM backend", ["openai", "local"])
    retriever = st.sidebar.selectbox(
        "Retriever", ["rerank", "hybrid_rrf", "dense"]
    )
    top_k = st.sidebar.slider("top_k", 1, 8, 4)

    key = f"chain_{backend}_{retriever}_{top_k}"
    if key not in st.session_state:
        with st.spinner("Loading index and models..."):
            st.session_state[key] = build_chain(backend, retriever, top_k)
    chain = st.session_state[key]

    question = st.text_input("Ask a question about the FAQ:")
    if question:
        with st.spinner("Retrieving..."):
            results = chain.retrieve(question)
        st.subheader("Retrieved documents")
        if not results:
            st.info("No documents retrieved.")
        else:
            for i, r in enumerate(results, 1):
                st.markdown(
                    f"**Passage {i}** (section: {r.chunk.section}, score: {r.score:.3f})"
                )
                st.text(r.chunk.text)

        with st.spinner("Thinking..."):
            resp = chain.generate(question, results)
        st.subheader("Answer")
        st.markdown(resp.answer)


def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:  # noqa: BLE001
        return False


if _in_streamlit():
    run_streamlit()
elif __name__ == "__main__":
    run_cli()

