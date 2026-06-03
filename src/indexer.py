"""
Incremental Indexing Manifest
------------------------------
Tracks what has already been embedded & stored in ChromaDB.
On subsequent runs with the same bookmark position, skips re-embedding entirely.
On bookmark advance, triggers a full re-index (delta indexing is a future enhancement).
"""

import json
import os
from datetime import datetime

MANIFEST_FILE = "data/index_manifest.json"


def load_manifest() -> dict | None:
    """Returns the manifest dict, or None if no manifest exists (first run)."""
    if not os.path.exists(MANIFEST_FILE):
        return None
    try:
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_manifest(bookmark: dict, memory_count: int, embedding_backend: str = "sentence-transformers"):
    """Writes a new manifest after a successful index run."""
    manifest = {
        "last_indexed_pov": bookmark["pov"],
        "last_indexed_occurrence": bookmark["occurrence"],
        "memory_count": memory_count,
        "embedding_backend": embedding_backend,
        "indexed_at": datetime.today().strftime("%Y-%m-%d")
    }
    os.makedirs(os.path.dirname(MANIFEST_FILE), exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
    return manifest


def needs_reindex(bookmark: dict, manifest: dict | None, current_backend: str = "sentence-transformers") -> tuple[bool, str]:
    """
    Returns (should_reindex: bool, reason: str).

    Reasons:
      "skip"    — bookmark unchanged, same backend → use existing ChromaDB collection
      "full"    — first run, or backend changed, or bookmark moved
    """
    if manifest is None:
        return True, "full"  # First run

    backend_changed = manifest.get("embedding_backend", "") != current_backend
    if backend_changed:
        return True, "full"  # Embedding space changed; must rebuild

    position_changed = (
        manifest["last_indexed_pov"] != bookmark["pov"] or
        manifest["last_indexed_occurrence"] != bookmark["occurrence"]
    )
    if position_changed:
        return True, "full"  # Bookmark advanced; rebuild with new text slice

    return False, "skip"
