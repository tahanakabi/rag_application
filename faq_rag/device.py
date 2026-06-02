"""Torch device selection helpers (GPU when available)."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def resolve_device(preference: str = "auto") -> str:
    """Return a torch device string.

    ``preference`` may be ``"auto"`` (CUDA -> MPS -> CPU), or an explicit
    device such as ``"cuda"``, ``"cuda:0"``, ``"mps"`` or ``"cpu"``.
    """
    pref = (preference or "auto").lower()
    if pref != "auto":
        return pref

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 - torch missing / probe failure -> CPU
        pass
    return "cpu"

