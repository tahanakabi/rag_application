"""Evaluate the RAG application end-to-end with OpenAI and the local LLM.

Logs generation metrics (ragas if available, else lexical proxies) to MLflow.

Usage:
    python -m scripts.run_generation_eval --backend openai --n 15
    python -m scripts.run_generation_eval --backend local  --n 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow

from faq_rag.config import get_settings
from faq_rag.evaluation.eval_set import build_eval_set, load_eval_set, save_eval_set
from faq_rag.evaluation.generation_metrics import GenSample, evaluate_generation
from faq_rag.models import load_chunks
from faq_rag.pipeline import build_all_retrievers, build_embedder, load_store_and_vectors
from faq_rag.rag.chain import RagChain


def _build_llm(backend: str, settings):
    if backend == "openai":
        from faq_rag.rag.llm_openai import OpenAILLM

        return OpenAILLM(model=settings.openai_model, api_key=settings.openai_api_key)
    from faq_rag.rag.llm_local import LocalLLM

    return LocalLLM(repo_id=settings.local_llm_repo, filename=settings.local_llm_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["openai", "local"], default="local")
    parser.add_argument("--retriever", default="rerank")
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    settings = get_settings()
    chunks = load_chunks(settings.chunks_file)

    eval_path = settings.data_path / "eval_set.json"
    examples = load_eval_set(eval_path) if eval_path.exists() else build_eval_set(chunks)
    if not eval_path.exists():
        save_eval_set(examples, eval_path)
    # de-duplicate by query and cap to n
    seen, picked = set(), []
    for ex in examples:
        if ex.query not in seen:
            seen.add(ex.query)
            picked.append(ex)
        if len(picked) >= args.n:
            break

    embedder = build_embedder(settings)
    store, vectors = load_store_and_vectors(settings, chunks, embedder)
    retrievers = build_all_retrievers(
        settings, chunks, embedder, store, full_vectors=vectors
    )
    retriever = retrievers[args.retriever]

    llm = _build_llm(args.backend, settings)
    chain = RagChain(retriever, llm, top_k=args.top_k)

    samples: list[GenSample] = []
    print(f"Generating answers with {llm.name} ...")
    for ex in picked:
        resp = chain.answer(ex.query)
        samples.append(
            GenSample(question=ex.query, answer=resp.answer, contexts=resp.contexts)
        )
        print(f"  Q: {ex.query[:70]}")
        print(f"  A: {resp.answer[:100]}\n")

    metrics = evaluate_generation(samples, openai_api_key=settings.openai_api_key)

    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment("faq_generation")
    with mlflow.start_run(run_name=f"{args.backend}_{args.retriever}"):
        mlflow.log_params(
            {
                "backend": args.backend,
                "llm": llm.name,
                "retriever": args.retriever,
                "top_k": args.top_k,
                "n_samples": len(samples),
            }
        )
        mlflow.log_metrics(metrics)

    print("Generation metrics:", metrics)
    print(f"MLflow runs logged to {settings.mlflow_uri}")


if __name__ == "__main__":
    main()

