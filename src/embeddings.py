"""
Embedding Backend
-----------------
Supports two backends, selected automatically:

  1. sentence-transformers + MPS (Apple M3 Metal Performance Shaders)
     → Routes through the M3 Neural Engine. ~3-5x faster than CPU.
     → Activated when `torch.backends.mps.is_available()` is True.
     NOTE: nomic-embed-text-v1 uses trust_remote_code which initialises
           a gRPC runtime. This deadlocks when another gRPC process is
           running (Ollama, IDE runner). Use MPS only from a clean terminal.

  2. Ollama (default/fallback)
     → Uses the local Ollama server with nomic-embed-text.
     → Activated when MPS is not available or USE_MPS = False.

Both backends use `nomic-ai/nomic-embed-text-v1`, producing 768-dim vectors.

Disk Cache
----------
Every embedding is cached to .embedding_cache/ keyed by SHA-256 of the text.
This means each unique sentence is only ever embedded ONCE — even across
multiple ablation configs or restarts. On the ablation study, configs C1-C7
share ~90% of sentences with C0, so they run in seconds from cache.
"""

import os
import json
import hashlib

# ── Backend config ────────────────────────────────────────────────────────────
USE_MPS = False   # nomic-embed-text gRPC conflicts with Ollama/IDE; keep False unless running from a clean terminal with Ollama stopped.

# ── Embedding disk cache ──────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".embedding_cache")
_mem_cache: dict = {}   # In-process cache (survives multiple calls in same run)


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_from_cache(text: str):
    """Returns cached embedding vector or None."""
    key = _cache_key(text)
    # 1. Check in-process memory first (fastest)
    if key in _mem_cache:
        return _mem_cache[key]
    # 2. Check disk cache
    path = os.path.join(CACHE_DIR, key[:2], key + ".json")
    if os.path.exists(path):
        with open(path, "r") as f:
            vec = json.load(f)
        _mem_cache[key] = vec
        return vec
    return None


def _save_to_cache(text: str, vec: list):
    """Persists an embedding vector to disk cache."""
    key = _cache_key(text)
    _mem_cache[key] = vec
    os.makedirs(os.path.join(CACHE_DIR, key[:2]), exist_ok=True)
    path = os.path.join(CACHE_DIR, key[:2], key + ".json")
    with open(path, "w") as f:
        json.dump(vec, f)


# ── Model management ──────────────────────────────────────────────────────────
_LOAD_FAILED = object()   # Sentinel: prevents retry loops on model load failure
_st_model = None          # Lazy-loaded sentence-transformers model


def _get_st_model():
    """
    Disabled for IDE runner to prevent gRPC/MPS deadlocks.
    Always falls back to Ollama natively.
    """
    return None

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


def _embed_uncached(text: str) -> list:
    """Calls the actual backend (no cache layer)."""
    try:
        return _embed_via_st(text)
    except Exception:
        pass  # Fall through to Ollama
    return _embed_via_ollama(text)


def _embed(text: str) -> list:
    """Embeds a single string with cache."""
    cached = _load_from_cache(text)
    if cached is not None:
        return cached
    vec = _embed_uncached(text)
    _save_to_cache(text, vec)
    return vec


# ── Public API ────────────────────────────────────────────────────────────────

def embed_memories(memories: list, model: str = "nomic-embed-text") -> list:
    """
    Embeds a list of memory strings with disk + in-process caching.

    Cache hits are instant (microseconds). Only genuinely new strings
    hit the Ollama/MPS backend. On repeated ablation runs or across
    configs that share sentences, this is dramatically faster.
    """
    assert isinstance(memories, list)
    assert all(isinstance(m, str) for m in memories)

    # Split into cache hits and misses
    results = [None] * len(memories)
    misses = []   # (original_index, text)

    for i, text in enumerate(memories):
        cached = _load_from_cache(text)
        if cached is not None:
            results[i] = cached
        else:
            misses.append((i, text))

    if misses:
        miss_texts = [t for _, t in misses]

        # Use batch ST encoding if model loads
        st_model = _get_st_model()
        if st_model is not None:
            print(f"[Embeddings] Batch-embedding {len(miss_texts)} new memories via MPS...")
            vecs = st_model.encode(miss_texts, normalize_embeddings=True, show_progress_bar=False)
            miss_vecs = [v.tolist() for v in vecs]
        else:
            # Ollama — sequential but with progress
            hits = len(memories) - len(misses)
            print(f"[Embeddings] {hits} cache hits, {len(miss_texts)} new → Ollama...")
            miss_vecs = [_embed_via_ollama(t) for t in miss_texts]

        # Store results and persist to cache
        for (orig_idx, text), vec in zip(misses, miss_vecs):
            results[orig_idx] = vec
            _save_to_cache(text, vec)
    else:
        print(f"[Embeddings] All {len(memories)} memories served from cache ⚡")

    assert all(r is not None for r in results)
    dim = len(results[0])
    assert all(len(v) == dim for v in results)
    return results


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


def cache_stats() -> dict:
    """Returns stats about the embedding cache."""
    if not os.path.exists(CACHE_DIR):
        return {"entries": 0, "size_mb": 0}
    count = sum(len(files) for _, _, files in os.walk(CACHE_DIR))
    size = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(CACHE_DIR)
        for f in files
    )
    return {"entries": count, "size_mb": round(size / 1024 / 1024, 1)}
