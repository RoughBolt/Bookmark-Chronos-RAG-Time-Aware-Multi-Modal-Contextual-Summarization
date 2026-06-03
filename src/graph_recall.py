"""
G-RAG: Graph-Augmented Retrieval
----------------------------------
Extends flat vector retrieval with 1-hop neighborhood expansion
over the character co-occurrence graph stored in data/knowledge.json.

After standard vector recall returns top-k results, this module:
  1. Identifies which characters appear in the recalled memories.
  2. Traverses the interaction graph to find high-weight neighbor characters.
  3. Queries ChromaDB for memories featuring those neighbors.
  4. Returns an augmented, deduplicated result set.

Reference: Architecturally analogous to Microsoft GraphRAG's community-based
retrieval, using co-occurrence edge weights as the graph signal.
"""

import json
import os
from src.knowledge.characters import extract_characters
from src.embeddings import embed_text


def graph_augment(
    recalled_results: list,
    collection,
    embed_fn=None,
    knowledge_file: str = "data/knowledge.json",
    top_neighbors: int = 3,
    max_augment: int = 2
) -> list:
    """
    Augments a set of recalled memory results with graph-neighborhood memories.

    Args:
        recalled_results:  List of dicts from recall_memories().
        collection:        Active ChromaDB collection.
        embed_fn:          Callable to embed a string (defaults to embed_text).
        knowledge_file:    Path to the knowledge graph JSON.
        top_neighbors:     How many neighbor characters to expand to (default 3).
        max_augment:       Max new memories to add per neighbor character (default 2).

    Returns:
        Augmented list of memory dicts (original + neighborhood memories).
    """
    if embed_fn is None:
        embed_fn = embed_text

    if not recalled_results or not os.path.exists(knowledge_file):
        return recalled_results

    # ── Step 1: Who is in the recalled memories? ──────────────────────────────
    local_chars: set = set()
    for r in recalled_results:
        chars = extract_characters(r.get("text", ""))
        local_chars.update(chars)

    if not local_chars:
        return recalled_results

    # ── Step 2: Load interaction graph, find high-weight neighbors ────────────
    try:
        with open(knowledge_file, "r", encoding="utf-8") as f:
            kg = json.load(f)
    except Exception:
        return recalled_results

    interactions: dict = kg.get("interactions", {})
    neighbor_weights: dict = {}

    for char in local_chars:
        for neighbor, weight in interactions.get(char, {}).items():
            if neighbor not in local_chars:  # Don't re-fetch already-retrieved chars
                prev = neighbor_weights.get(neighbor, 0)
                neighbor_weights[neighbor] = max(prev, weight)

    if not neighbor_weights:
        return recalled_results

    # Sort neighbors by edge weight (strongest relationship first)
    sorted_neighbors = sorted(neighbor_weights.items(), key=lambda x: x[1], reverse=True)
    expansion_chars = [name for name, _ in sorted_neighbors[:top_neighbors]]

    # ── Step 3: Query ChromaDB for memories containing each neighbor ──────────
    existing_texts = {r.get("text", "")[:60] for r in recalled_results}
    augmented = list(recalled_results)

    for char_name in expansion_chars:
        try:
            query_vec = embed_fn(char_name)
            results = collection.query(
                query_embeddings=[query_vec],
                n_results=max_augment + 2,  # fetch extra, filter below
                include=["documents", "metadatas", "distances"]
            )

            added = 0
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                if added >= max_augment:
                    break
                fingerprint = doc[:60]
                if fingerprint in existing_texts:
                    continue
                # Only add if reasonably relevant (not too far)
                if dist > 0.65:
                    continue

                augmented.append({
                    "text": doc,
                    "tag": meta.get("tag", "EVENT"),
                    "index": meta.get("index", 0),
                    "distance": dist,
                    "source": f"graph_augment({char_name})"
                })
                existing_texts.add(fingerprint)
                added += 1

        except Exception:
            continue

    # Restore chronological order after augmentation
    augmented.sort(key=lambda x: x.get("index", 0))
    return augmented
