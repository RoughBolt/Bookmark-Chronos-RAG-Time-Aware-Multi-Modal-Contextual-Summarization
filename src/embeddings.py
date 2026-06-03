"""
Embedding Backend
-----------------
Supports two backends, selected automatically:

  1. sentence-transformers + MPS (Apple M3 Metal Performance Shaders)
     → Routes through the M3 Neural Engine. ~3-5x faster than CPU.
     → Activated when `torch.backends.mps.is_available()` is True.

  2. Ollama (fallback)
     → Uses the local Ollama server with nomic-embed-text.
     → Activated when MPS is not available or USE_MPS = False.

Both backends use `nomic-ai/nomic-embed-text-v1`, producing 768-dim vectors.

To force Ollama (e.g., for debugging), set: USE_MPS = False
"""

import os

# ── Backend config ────────────────────────────────────────────────────────────
# Set to True once `pip install einops` is done and the model has been
# downloaded at least once (run: python3 -c "from src.embeddings import _get_st_model; _get_st_model()").
# Until then, keep False — Ollama handles all embeddings correctly.
USE_MPS = False   # ← flip to True to activate M3 MPS acceleration

_LOAD_FAILED = object()   # Sentinel: prevents retry loops on model load failure
_st_model = None          # Lazy-loaded sentence-transformers model


def _get_st_model():
    """
    Lazy-loads the sentence-transformers model with MPS or CPU device.
    Uses a sentinel (_LOAD_FAILED) to prevent repeated retry attempts
    if the model fails to load (e.g. not yet downloaded, missing einops).
    """
    global _st_model

    if _st_model is _LOAD_FAILED:
        return None  # Already failed once — don't retry, use Ollama

    if _st_model is None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "mps" if (USE_MPS and torch.backends.mps.is_available()) else "cpu"

            print(f"[Embeddings] Loading nomic-embed-text on device={device} ...")
            _st_model = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1",
                device=device,
                trust_remote_code=True,
                local_files_only=True   # Never block on network — use cache only
            )
            print(f"[Embeddings] Model ready (device={device})")

        except Exception as e:
            # Model not cached or einops missing — mark as failed, fall back to Ollama
            print(f"[Embeddings] sentence-transformers unavailable ({type(e).__name__}: {e})")
            print("[Embeddings] Falling back to Ollama (nomic-embed-text).")
            _st_model = _LOAD_FAILED
            return None

    return _st_model if _st_model is not _LOAD_FAILED else None


def _embed_via_st(text: str) -> list:
    model = _get_st_model()
    if model is None:
        raise RuntimeError("sentence-transformers not available")
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def _embed_via_ollama(text: str) -> list:
    import ollama
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def _embed(text: str) -> list:
    """Routes to MPS backend, falls back to Ollama if unavailable."""
    if USE_MPS:
        try:
            return _embed_via_st(text)
        except Exception:
            pass  # Fall through to Ollama
    return _embed_via_ollama(text)


# ── Public API ────────────────────────────────────────────────────────────────

def embed_memories(memories: list, model: str = "nomic-embed-text") -> list:
    """
    Embeds a list of memory strings.
    Uses batch encoding when sentence-transformers is available (faster on MPS).
    Falls back to sequential Ollama calls if ST model is not loaded.
    """
    assert isinstance(memories, list)
    assert all(isinstance(m, str) for m in memories)

    st_model = _get_st_model()
    if st_model is not None:
        # Batch encode — much faster on MPS than one-by-one
        vecs = st_model.encode(memories, normalize_embeddings=True, show_progress_bar=True)
        embeddings = [v.tolist() for v in vecs]
    else:
        # Ollama fallback — sequential (works without any extra deps)
        print(f"[Embeddings] Embedding {len(memories)} memories via Ollama...")
        embeddings = [_embed_via_ollama(m) for m in memories]

    assert len(embeddings) == len(memories)
    dim = len(embeddings[0])
    for vec in embeddings:
        assert len(vec) == dim

    return embeddings


def embed_text(text: str, model: str = "nomic-embed-text") -> list:
    """Embeds a single string. Used for query embedding and semantic chunking."""
    return _embed(text)


def active_backend() -> str:
    """Returns the name of the currently active embedding backend."""
    if USE_MPS:
        st = _get_st_model()
        if st is not None:
            try:
                import torch
                if torch.backends.mps.is_available():
                    return "sentence-transformers-mps"
                return "sentence-transformers-cpu"
            except ImportError:
                pass
    return "ollama"
