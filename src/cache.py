"""
Query Embedding Cache
---------------------
In-memory LRU cache for query embeddings, persisted to disk between sessions.
Prevents redundant Ollama/MPS calls for repeated queries.
"""

import json
import os
from collections import OrderedDict

CACHE_FILE = "data/embed_cache.json"
MAX_CACHE_SIZE = 100

_cache: OrderedDict = OrderedDict()
_cache_loaded: bool = False


def _load_cache():
    global _cache, _cache_loaded
    if _cache_loaded:
        return
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            _cache = OrderedDict(data)
        except Exception:
            _cache = OrderedDict()
    _cache_loaded = True


def _save_cache():
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(dict(_cache), f)


def cached_embed(text: str, embed_fn) -> list:
    """
    Returns a cached embedding for `text` if available.
    Otherwise calls `embed_fn(text)`, stores the result, and returns it.
    Evicts the oldest entry when the cache exceeds MAX_CACHE_SIZE.
    """
    _load_cache()

    key = text.strip()

    if key in _cache:
        return _cache[key]

    embedding = embed_fn(text)

    # LRU eviction — remove oldest entry
    if len(_cache) >= MAX_CACHE_SIZE:
        _cache.popitem(last=False)

    _cache[key] = embedding
    _save_cache()

    return embedding


def clear_cache():
    """Clears in-memory and on-disk cache."""
    global _cache, _cache_loaded
    _cache = OrderedDict()
    _cache_loaded = True
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)


def cache_stats() -> dict:
    """Returns current cache stats for profiling output."""
    _load_cache()
    return {"entries": len(_cache), "max_size": MAX_CACHE_SIZE}
