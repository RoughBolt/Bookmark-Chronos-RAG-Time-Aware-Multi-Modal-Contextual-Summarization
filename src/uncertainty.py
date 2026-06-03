"""
Perturbation-Based Retrieval Uncertainty Estimator
----------------------------------------------------
Addresses "hallucination-by-retrieval" — where the system is forced to answer
using irrelevant context. When uncertainty is high, the system should abstain
rather than produce a low-quality narrative.

Mechanism (Monte Carlo Perturbation proxy):
  1. Generate N perturbed versions of the query embedding by adding
     small Gaussian noise (σ=0.01).
  2. Retrieve top-k results for each perturbed embedding.
  3. Measure Jaccard similarity across the N result sets.
  4. uncertainty = 1 - mean_jaccard_similarity

Interpretation:
  - uncertainty ≈ 0.0 → all perturbations retrieve the same docs (high confidence)
  - uncertainty ≈ 1.0 → completely different docs each time (retrieval is unstable)

Threshold: uncertainty > 0.7 → return "Insufficient Context" instead of hallucinated narrative.

Note: True Monte Carlo Dropout requires access to model internals (dropout layers).
Since Ollama and sentence-transformers are black-box at inference time, this
perturbation-based proxy is the standard alternative used in production RAG systems.
"""

import random
import math


# ── Noise injection ───────────────────────────────────────────────────────────

def _perturb(embedding: list, sigma: float = 0.01) -> list:
    """Adds Gaussian noise to an embedding vector."""
    return [v + random.gauss(0, sigma) for v in embedding]


# ── Jaccard similarity on ranked result sets ──────────────────────────────────

def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


# ── Core uncertainty estimator ────────────────────────────────────────────────

def compute_retrieval_uncertainty(
    query_embedding: list,
    collection,
    n_passes: int = 5,
    sigma: float = 0.01,
    k: int = 5
) -> float:
    """
    Estimates retrieval uncertainty via embedding perturbation.

    Args:
        query_embedding:  Original query vector.
        collection:       Active ChromaDB collection.
        n_passes:         Number of noisy forward passes (default 5).
        sigma:            Gaussian noise std dev (default 0.01).
        k:                Number of results per pass (default 5).

    Returns:
        Uncertainty score in [0.0, 1.0].
        0.0 = perfectly stable retrieval (certain).
        1.0 = completely unstable retrieval (uncertain).
    """
    result_sets: list[set] = []

    for _ in range(n_passes):
        perturbed = _perturb(query_embedding, sigma)
        try:
            results = collection.query(
                query_embeddings=[perturbed],
                n_results=k,
                include=["documents"]
            )
            # Use first 40 chars of each doc as a fingerprint
            doc_ids = frozenset(d[:40] for d in results["documents"][0])
            result_sets.append(doc_ids)
        except Exception:
            continue

    if len(result_sets) < 2:
        return 0.0  # Cannot measure — assume certain

    # Compute pairwise Jaccard similarities
    pairwise = []
    for i in range(len(result_sets)):
        for j in range(i + 1, len(result_sets)):
            pairwise.append(_jaccard(result_sets[i], result_sets[j]))

    mean_jaccard = sum(pairwise) / len(pairwise)
    uncertainty = 1.0 - mean_jaccard

    return round(uncertainty, 3)


# ── Threshold guard ───────────────────────────────────────────────────────────

UNCERTAINTY_THRESHOLD = 0.70

def is_uncertain(score: float, threshold: float = UNCERTAINTY_THRESHOLD) -> bool:
    """Returns True if retrieval uncertainty exceeds the acceptable threshold."""
    return score > threshold
