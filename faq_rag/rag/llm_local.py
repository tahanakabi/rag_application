"""Small quantized local LLM backend via llama-cpp-python (GGUF, CPU-friendly).

The GGUF file is pulled from the Hugging Face hub on first use and cached.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _load_llama(repo_id: str, filename: str, n_ctx: int):
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return Llama(model_path=model_path, n_ctx=n_ctx, verbose=False)


class LocalLLM:
    """Quantized HF GGUF model run locally on CPU."""

    def __init__(
        self,
        repo_id: str = "bartowski/Qwen2.5-3B-Instruct-GGUF",
        filename: str = "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        n_ctx: int = 2048,
        temperature: float = 0.0,
        max_tokens: int = 384,
    ):
        self.name = f"local:{repo_id.split('/')[-1]}/{filename}"
        self.repo_id = repo_id
        self.filename = filename
        self.n_ctx = n_ctx
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = _load_llama(self.repo_id, self.filename, self.n_ctx)
        return self._llm

    def generate(self, system: str, user: str) -> str:
        # Prefer the model's built-in chat template; fall back to a raw prompt.
        try:
            out = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return (out["choices"][0]["message"]["content"] or "").strip()
        except Exception:  # noqa: BLE001 - some GGUFs lack a chat template
            prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
            out = self.llm(
                prompt, temperature=self.temperature, max_tokens=self.max_tokens,
                stop=["<|user|>", "<|system|>"],
            )
            return out["choices"][0]["text"].strip()

