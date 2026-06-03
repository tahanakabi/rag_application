# FAQ RAG Chatbot

An LLM-powered Retrieval-Augmented Generation (RAG) chatbot that answers
questions **strictly** from the FAQ document
[`taxonomy_faqs_cleaned.md`](taxonomy_faqs_cleaned.md). It covers the full
pipeline: markdown ingestion with OCR for embedded base64 images, a FAISS
vector store, three retrieval strategies benchmarked with MLflow,
and a context-constrained RAG chain backed by both an OpenAI model and a small
quantized local LLM, with generation metrics logged to MLflow.

## Architecture

```
faq_rag/
  config.py            # env/.env settings (pydantic-settings)
  models.py            # Chunk dataclass + JSONL persistence
  pipeline.py          # build embedder / stores / retrievers
  ingestion/
    parser.py          # split ## sections and ### Q/A pairs
    image_ocr.py       # decode base64 images -> OCR (tesseract/easyocr) + cache
    chunker.py         # long-answer sub-chunking, question kept as context
    embeddings.py      # sentence-transformers embeddings (L2-normalised)
  stores/
    faiss_store.py     # in-memory FAISS (cosine via inner product)
  retrieval/
    dense.py           # semantic baseline
    hybrid.py          # dense + BM25 fused with Reciprocal Rank Fusion
    rerank.py          # cross-encoder reranking
  evaluation/
    eval_set.py        # auto + paraphrase question->chunk gold set
    retrieval_metrics.py   # Hit@k, MRR, Recall@k, nDCG
    generation_metrics.py  # ragas faithfulness/relevancy/precision (+ lexical fallback)
  rag/
    prompts.py         # context-only system prompt + refusal
    llm_openai.py      # OpenAI backend (key from env)
    llm_local.py       # quantized GGUF model via llama-cpp-python
    chain.py           # retrieve -> prompt -> generate
  app.py               # CLI + Streamlit chatbot
scripts/
  run_ingest.py            # parse -> OCR -> chunk -> embed -> FAISS
  run_retrieval_eval.py    # benchmark retrievers -> MLflow
  run_generation_eval.py   # benchmark RAG answers -> MLflow
```

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# system dependency for OCR (Debian/Ubuntu)
sudo apt-get install -y tesseract-ocr

```
Create a `.env` file (see `.env.example`) with your OpenAI API key and desired model names. The default embedding and reranking models are small and run on CPU
by default, but will use the GPU if available (`DEVICE=auto`); you can also force a specific device with `DEVICE=cuda`, `DEVICE=mps` or `DEVICE=cpu`.
## Usage

### 1. Ingestion (chunking, OCR, embedding, vector DBs)
```bash
uv run scripts/run_ingest.py
```
Builds Q/A chunks (long answers split with the question preserved as context),
OCRs embedded base64 images, then stores embeddings in **FAISS** (in-memory,
persisted to disk).

### 2. Retrieval experiments (3 approaches + metrics)
Start an Mlflow server:

```bash
mlflow server \                                
    --backend-store-uri sqlite:///mlflow.db \           
    --default-artifact-root ./artifacts \                               
    --host 0.0.0.0
 ```

```bash
uv run scripts/run_retrieval_eval.py
```
Evaluates **dense**, **hybrid (RRF)** and **cross-encoder rerank** retrievers.
Metrics tracked: **Hit@k**, **MRR**, **Recall@k**,
**nDCG@k**. 

View runs at localhost:5000 to compare retriever performance. The best retriever (likely
### 3. RAG application + generation metrics
```bash
# OpenAI backend
uv run scripts/run_generation_eval.py --backend openai --retriever rerank

# Small quantized local model (tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf)
uv run scripts/run_generation_eval.py --backend local --retriever rerank
```
Generation metrics (faithfulness, answer relevancy, context precision via
**ragas**; lexical proxies if ragas/judge LLM unavailable) are logged to MLflow.

### Chat interactively
```bash
uv run faq_rag/app.py --backend openai --retriever rerank   # CLI
uv run streamlit run faq_rag/app.py                                # web UI
```

## Design notes / choices

- **Chunking** follows the natural question/answer split; answers exceeding a
  token budget (~384) are split into overlapping sub-chunks, each prepended with
  its question so context is never lost.
- **Images**: base64 PNGs are decoded, OCR'd (tesseract by default, easyocr
  optional), cached by content hash, and their text appended to the chunk; the
  heavy markup is stripped before embedding.
- **Tracked metric**: retrieval quality is primarily compared via **nDCG@k /
  MRR**, demonstrating how hybrid + reranking improve over the dense baseline;
  generation is compared via **faithfulness** to show the quantized vs. OpenAI
  trade-off. All experiments are logged to **MLflow**.
- **Grounding**: the system prompt forbids outside knowledge and the chain can
  refuse (`min_score`) when retrieval confidence is low, keeping answers scoped
  to the FAQ only.

## Configuration
All knobs live in `.env` (see `.env.example`): embedding/reranker/LLM model
names, OCR engine, and data paths. `OPENAI_API_KEY` is
read from the environment. The embedding and reranking models run on the GPU
automatically when available (`DEVICE=auto`, CUDA → MPS → CPU); force a device
with `DEVICE=cuda`, `DEVICE=mps` or `DEVICE=cpu`.

