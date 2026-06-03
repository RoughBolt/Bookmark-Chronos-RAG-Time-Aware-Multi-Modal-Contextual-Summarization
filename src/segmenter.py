"""
Segmenter — Semantic Chunking
------------------------------
Splits text into topically coherent chunks using cosine similarity drift detection.
Two consecutive sentences with similarity < threshold signal a topic boundary.

Falls back to length-based splitting if embedding is unavailable.
"""

import re


# ── Cosine similarity (no numpy dependency) ──────────────────────────────────

def _cosine(v1: list, v2: list) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = sum(a * a for a in v1) ** 0.5
    n2 = sum(b * b for b in v2) ** 0.5
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


# ── Semantic chunking (primary method) ───────────────────────────────────────

def semantic_chunk(text: str, threshold: float = 0.75, min_sentences: int = 3) -> list[str]:
    """
    Splits `text` into semantically coherent chunks.

    Algorithm:
      1. Sentence-tokenize the text.
      2. Embed each sentence using the active embedding backend.
      3. Compute cosine similarity between consecutive sentence embeddings.
      4. Cut a new chunk when similarity drops below `threshold`,
         provided the current chunk already has >= `min_sentences` sentences.

    Args:
        text:          Full text to chunk.
        threshold:     Cosine similarity below which a topic shift is declared (default 0.75).
        min_sentences: Minimum sentences per chunk before a cut is allowed (default 3).

    Returns:
        List of chunk strings (joined sentences per chunk).
    """


    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) <= min_sentences:
        return [text]

    try:
        from src.embeddings import embed_memories as _embed_batch
        print(f"[Chunker] Batch-embedding {len(sentences)} sentences for semantic boundary detection...")
        # Batch embed all sentences at once (much faster than sequential embed_text calls)
        embeddings = _embed_batch(sentences)

        chunks = []
        current: list[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = _cosine(embeddings[i - 1], embeddings[i])

            # Cut only if we have enough sentences AND similarity dropped
            if sim < threshold and len(current) >= min_sentences:
                chunks.append(" ".join(current))
                current = [sentences[i]]
            else:
                current.append(sentences[i])

        if current:
            chunks.append(" ".join(current))

        print(f"[Chunker] {len(sentences)} sentences → {len(chunks)} semantic chunks (threshold={threshold})")
        return chunks

    except Exception as e:
        print(f"[Chunker] Embedding unavailable ({e}), falling back to length-based chunking.")
        return split_into_paragraphs(text)


# ── Length-based chunking (fallback) ─────────────────────────────────────────

def split_into_paragraphs(text: str, min_length: int = 200) -> list[str]:
    """
    Fallback: splits text into paragraph-like chunks by character count.
    Ensures chunks are not too small.
    """
    raw_chunks = text.split(". ")
    paragraphs = []
    current = []

    for sentence in raw_chunks:
        current.append(sentence)
        if sum(len(s) for s in current) >= min_length:
            paragraphs.append(". ".join(current).strip() + ".")
            current = []

    if current:
        paragraphs.append(". ".join(current).strip() + ".")

    return paragraphs