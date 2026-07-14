"""CPU-only in-process embedder for all-MiniLM-L6-v2 (384d) via fastembed/onnxruntime.

House-style (spec-python-ml-retrieval): embedding is the SEMANTIC fallback for
fuzzy markdown-fact recall; a structural frontmatter filter scopes the candidate
set first (this POC exercises the dense leg only). The model loads ONCE and runs
the blocking ONNX encode synchronously — callers on an async loop must offload it.

Why fastembed (measured, recorded for the brief's "which backend won and why"):
fastembed runs MiniLM as an optimized/quantized ONNX graph on onnxruntime's
CPUExecutionProvider and pulls NO torch. sentence-transformers with
backend="onnx" produces the same 384-d vectors but drags the full torch stack
into the shared venv — which both bloats phi's host and risks a CUDA path. d3
forbids GPU contention with phi4-mini, so fastembed is the lean, CPU-pinned win.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

# LAZY fastembed import (live 2026-07-13 catch): importing fastembed pulls
# onnxruntime's native DLL, whose load was measured at up to ~24 MINUTES on this
# host (onnxruntime.capi._pybind_state self-time in -X importtime) — and this
# module was imported EAGERLY by memory → chat_app.app → chat_app, so EVERY
# process (each pytest session, the app boot, every probe) paid it, even though
# most never embed a single text. fastembed now loads on FIRST USE only; the
# module import is instant.
TextEmbedding = None  # populated by _load_fastembed on first use


def _load_fastembed():
    global TextEmbedding
    if TextEmbedding is None:
        from fastembed import TextEmbedding as _TE
        TextEmbedding = _TE
    return TextEmbedding


# all-MiniLM-L6-v2: 384-dimensional sentence embeddings.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384
# Pin CPU explicitly so a stray onnxruntime-gpu install can never silently move
# embedding onto CUDA and start contending with the resident chat model (d3).
CPU_PROVIDERS = ["CPUExecutionProvider"]


class CpuEmbedder:
    """Loads MiniLM once on CPU and embeds text to L2-normalized 384-d float32."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        # fastembed defaults to CPU, but pin providers so the choice is explicit
        # and auditable rather than implicit. Imported lazily — see _load_fastembed.
        self._model = _load_fastembed()(model_name=model_name, providers=CPU_PROVIDERS)

    @property
    def providers(self) -> list[str]:
        """The onnxruntime execution providers this embedder is pinned to."""
        return list(CPU_PROVIDERS)

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        """Embed an iterable of strings → ndarray of shape (n, 384), float32.

        fastembed yields per-text float32 vectors (already L2-normalized for
        all-MiniLM); we stack them into a single matrix for the store.
        """
        items = list(texts)
        if not items:
            return np.empty((0, DIM), dtype=np.float32)
        vecs = np.asarray(list(self._model.embed(items)), dtype=np.float32)
        if vecs.shape[1] != DIM:  # fail fast at the boundary
            raise ValueError(f"expected {DIM}-d vectors, got {vecs.shape[1]}-d")
        return vecs

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single query string → ndarray of shape (384,), float32."""
        return self.embed([text])[0]
