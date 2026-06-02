"""Centralised configuration loaded from environment / .env file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = parent of the faq_rag package directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The source FAQ document
FAQ_DOC = PROJECT_ROOT / "taxonomy_faqs_cleaned.md"


class Settings(BaseSettings):
    """Application settings; values come from the environment or a .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Secrets / models
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    embed_model: str = "BAAI/bge-base-en-v1.5"
    embed_trust_remote_code: bool = False

    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Compute device for the embedding / reranking models:
    # "auto" picks CUDA -> MPS -> CPU; or set "cuda", "cuda:0", "mps", "cpu".
    device: str = "auto"

    local_llm_repo: str = "bartowski/Qwen2.5-3B-Instruct-GGUF"
    local_llm_file: str = "Qwen2.5-3B-Instruct-Q4_K_M.gguf"

    ocr_engine: str = "tesseract"

    # Paths (relative to project root unless absolute)
    data_dir: str = "data"
    faiss_index_path: str = "data/faiss.index"
    chunks_path: str = "data/chunks.jsonl"
    mlflow_tracking_uri: str = "data/mlruns"

    # ---- helpers to resolve paths relative to the project root ----
    def _resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def data_path(self) -> Path:
        path = self._resolve(self.data_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def faiss_index(self) -> Path:
        return self._resolve(self.faiss_index_path)

    @property
    def chunks_file(self) -> Path:
        return self._resolve(self.chunks_path)

    @property
    def mlflow_uri(self) -> str:
        path = self._resolve(self.mlflow_tracking_uri)
        path.mkdir(parents=True, exist_ok=True)
        return path.as_uri()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

