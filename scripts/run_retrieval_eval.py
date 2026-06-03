"""Evaluate the 3+ retrieval strategies and log metrics to MLflow.

Usage:
    python -m scripts.run_retrieval_eval --k 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow

from faq_rag.config import get_settings
from faq_rag.evaluation.eval_set import build_eval_set, load_eval_set, save_eval_set
from faq_rag.evaluation.retrieval_metrics import evaluate_retriever
from faq_rag.models import load_chunks
from faq_rag.pipeline import build_all_retrievers, build_embedder, load_store_and_vectors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n-eval", type=int, default=100)
    args = parser.parse_args()

    settings = get_settings()
    chunks = load_chunks(settings.chunks_file)
    print(f"Loaded {len(chunks)} chunks")

    # gold set (built once, cached on disk)
    eval_path = settings.data_path / "eval_set.json"
    if eval_path.exists():
        examples = load_eval_set(eval_path)
    else:
        examples = build_eval_set(chunks, n=args.n_eval)
        save_eval_set(examples, eval_path)
    print(f"Eval set: {len(examples)} queries")

    embedder = build_embedder(settings)
    store, vectors = load_store_and_vectors(settings, chunks, embedder)
    retrievers = build_all_retrievers(
        settings, chunks, embedder, store, full_vectors=vectors
    )

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("faq_retrieval")

    print(f"\n{'retriever':<14} hit_at_k   mrr     recall  ndcg")
    for name, retriever in retrievers.items():
        metrics = evaluate_retriever(retriever, examples, k=args.k)
        with mlflow.start_run(run_name=f"{name}"):
            mlflow.log_params(
                {
                    "retriever": name,
                    "embed_model": settings.embed_model,
                    "rerank_model": settings.rerank_model,
                    "k": args.k,
                    "n_queries": metrics.n_queries,
                }
            )
            mlflow.log_metrics(metrics.as_dict())
        print(
            f"{name:<14} {metrics.hit_at_k:.3f}   {metrics.mrr:.3f}   "
            f"{metrics.recall_at_k:.3f}   {metrics.ndcg_at_k:.3f}"
        )

    print(f"\nMLflow runs logged to http://localhost:5000")


if __name__ == "__main__":
    main()

