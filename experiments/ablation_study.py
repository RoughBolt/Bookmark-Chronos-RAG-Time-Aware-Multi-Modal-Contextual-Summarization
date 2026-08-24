"""
Ablation Study: Component-Level Impact Analysis
===================================================
Systematically evaluates the contribution of each Chronos-RAG component
by progressively adding them and measuring downstream retrieval quality.

Configurations tested (8 total):
  ┌────┬─────────────────────────────────────────────────────────────────┐
  │ ID │ Configuration                                                   │
  ├────┼─────────────────────────────────────────────────────────────────┤
  │ C0 │ Vanilla RAG (sentence splitting, embed, retrieve)              │
  │ C1 │ + Semantic Chunking                                             │
  │ C2 │ + Rule-Based Event Classification                               │
  │ C3 │ + Temporal Decay (single-scale, hand-tuned λ)                   │
  │ C4 │ + Temporal Decay (multi-scale, Atkinson-Shiffrin model)         │
  │ C5 │ + Knowledge Graph Augmentation (G-RAG)                          │
  │ C6 │ + Semantic Deduplication                                         │
  │ C7 │ Full Chronos-RAG (all components)                               │
  └────┴─────────────────────────────────────────────────────────────────┘

Metrics per configuration:
  - Keyword Recall (fast proxy for context_recall)
  - Precision@5 (fraction of retrieved docs containing ground-truth keywords)
  - Memory Count (how many memories were indexed)
  - Index Time (wall-clock time for pipeline execution)

Output:
  - experiments/ablation_results.json     — raw results
  - experiments/ablation_comparison.png   — bar chart comparison
  - experiments/ablation_table.md         — markdown table for papers/README

Usage:
  python -m experiments.ablation_study
  python -m experiments.ablation_study --configs C0 C3 C7   # run specific configs
  python -m experiments.ablation_study --days-gap 30         # test with 30-day gap

Requirements:
  pip install matplotlib (for plots)
"""

import argparse
import copy
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BOOK_PATH = "data/book.txt"
QA_PATH = "benchmark/qa_pairs.json"
OUTPUT_DIR = "experiments"


# ── Keyword recall metric (same as optimize_lambda.py) ────────────────────────

STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "are", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "and", "or",
    "but", "if", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "that", "this", "these", "those", "it",
    "its", "they", "them", "their", "he", "she", "him", "her", "his",
    "we", "our", "you", "your", "who", "what", "which", "when", "where",
    "how", "not", "no", "nor", "than", "too", "very", "just", "about",
    "also", "then", "so", "such", "both", "each", "other", "some", "any",
}


def extract_keywords(text: str, min_length: int = 3) -> set:
    words = re.findall(r'[a-z]+', text.lower())
    return {w for w in words if len(w) >= min_length and w not in STOPWORDS}


def keyword_recall(retrieved: list, ground_truth: str) -> float:
    gt_kw = extract_keywords(ground_truth)
    if not gt_kw:
        return 1.0
    ctx_kw = extract_keywords(" ".join(retrieved))
    return len(gt_kw & ctx_kw) / len(gt_kw)


def precision_at_k(retrieved: list, ground_truth: str, k: int = 5) -> float:
    gt_kw = extract_keywords(ground_truth)
    if not gt_kw:
        return 1.0
    hits = 0
    for doc in retrieved[:k]:
        doc_kw = extract_keywords(doc)
        if gt_kw & doc_kw:
            hits += 1
    return hits / min(k, len(retrieved)) if retrieved else 0.0


def _cosine_sim(v1: list, v2: list) -> float:
    """Cosine similarity between two vectors (no numpy dependency)."""
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = sum(a * a for a in v1) ** 0.5
    n2 = sum(b * b for b in v2) ** 0.5
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def semantic_recall(retrieved: list, ground_truth: str, embed_fn) -> float:
    """
    Computes semantic recall: max cosine similarity between
    the ground truth embedding and each retrieved context embedding.

    This is superior to keyword recall when ground truths use
    summary language that differs from the source text vocabulary
    (e.g., 'White Walker' vs 'the Others').
    """
    if not retrieved:
        return 0.0
    gt_vec = embed_fn(ground_truth)
    max_sim = 0.0
    for doc in retrieved:
        doc_vec = embed_fn(doc)
        sim = _cosine_sim(gt_vec, doc_vec)
        max_sim = max(max_sim, sim)
    return max_sim


# ── Pipeline configurations ──────────────────────────────────────────────────

def get_text(bookmark=None):
    """Load and preprocess book text."""
    from src.preprocessing import clean_text
    from src.utils import find_chapters, extract_text_upto_chapter
    from src.bookmark import load_bookmark

    bm = bookmark or load_bookmark()
    chapters = find_chapters(BOOK_PATH)
    text = extract_text_upto_chapter(BOOK_PATH, chapters, bm["pov"], bm["occurrence"])
    return clean_text(text)


def build_config_C0(text: str, days_gap: int) -> list:
    """C0: Vanilla RAG — sentence splitting, no event classification, no decay."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]
    return sentences


def build_config_C1(text: str, days_gap: int) -> list:
    """C1: + Semantic chunking (replaces naive sentence splitting)."""
    from src.segmenter import semantic_chunk
    chunks = semantic_chunk(text, threshold=0.75, min_sentences=3)
    return chunks


def build_config_C2(text: str, days_gap: int) -> list:
    """C2: + Rule-based event classification (extracts typed events from chunks)."""
    from src.segmenter import semantic_chunk
    from src.events import extract_events

    chunks = semantic_chunk(text, threshold=0.75, min_sentences=3)
    memories = []
    for chunk in chunks:
        events = extract_events(chunk)
        if events:
            memories.append(chunk)

    return memories


def build_config_C3(text: str, days_gap: int) -> list:
    """C3: + Single-scale temporal decay (hand-tuned λ per type)."""
    from src.segmenter import semantic_chunk
    from src.events import extract_events, apply_temporal_decay, get_event_threshold

    chunks = semantic_chunk(text, threshold=0.75, min_sentences=3)
    chunk_events_map = []
    all_events = []
    offset = 0
    for chunk in chunks:
        events = extract_events(chunk)
        for e in events:
            e["position"] += offset
        chunk_events_map.append((chunk, events))
        all_events.extend(events)
        offset += max((e["position"] for e in events), default=0) + 1

    # Force single-scale
    apply_temporal_decay(all_events, force_single_scale=True)

    scores = [e.get("decay_score", e["importance"]) for e in all_events]
    score_threshold = sorted(scores)[int(len(scores) * 0.25)] if scores else 0
    
    memories = []
    for chunk_text, events in chunk_events_map:
        surviving = [e for e in events if e.get("decay_score", e["importance"]) >= score_threshold]
        if surviving:
            memories.append(chunk_text)
            
    return memories


def build_config_C4(text: str, days_gap: int) -> list:
    """C4: + Multi-scale temporal decay (Atkinson-Shiffrin model)."""
    from src.segmenter import semantic_chunk
    from src.events import extract_events, get_event_threshold
    from src.temporal.multi_scale_decay import apply_multi_scale_decay

    chunks = semantic_chunk(text, threshold=0.75, min_sentences=3)
    chunk_events_map = []
    all_events = []
    offset = 0
    for chunk in chunks:
        events = extract_events(chunk)
        for e in events:
            e["position"] += offset
        chunk_events_map.append((chunk, events))
        all_events.extend(events)
        offset += max((e["position"] for e in events), default=0) + 1

    apply_multi_scale_decay(all_events)

    # Filter on decay_score, keeping top 75% by decayed importance.
    scores = [e.get("decay_score", e["importance"]) for e in all_events]
    score_threshold = sorted(scores)[int(len(scores) * 0.25)] if scores else 0
    
    memories = []
    for chunk_text, events in chunk_events_map:
        surviving = [e for e in events if e.get("decay_score", e["importance"]) >= score_threshold]
        if surviving:
            memories.append(chunk_text)
            
    return memories


def build_config_C5(text: str, days_gap: int) -> list:
    """C5: + Knowledge graph augmentation (G-RAG)."""
    from src.segmenter import semantic_chunk
    from src.events import extract_events, get_event_threshold
    from src.temporal.multi_scale_decay import apply_multi_scale_decay
    from src.knowledge.graph_builder import build_global_knowledge_graph

    chunks = semantic_chunk(text, threshold=0.75, min_sentences=3)
    chunk_events_map = []
    all_events = []
    offset = 0
    for chunk in chunks:
        events = extract_events(chunk)
        for e in events:
            e["position"] += offset
        chunk_events_map.append((chunk, events))
        all_events.extend(events)
        offset += max((e["position"] for e in events), default=0) + 1

    # Build knowledge graph
    kg = build_global_knowledge_graph(all_events, output_path="data/knowledge_ablation.json")

    apply_multi_scale_decay(all_events)

    # Filter on decay_score, keeping top 75% by decayed importance.
    scores = [e.get("decay_score", e["importance"]) for e in all_events]
    score_threshold = sorted(scores)[int(len(scores) * 0.25)] if scores else 0

    dual_channel_memories = []
    for chunk_text, events in chunk_events_map:
        surviving_events = [e for e in events if e.get("decay_score", e["importance"]) >= score_threshold]
        
        if not surviving_events:
            continue
            
        context_lines = []
        for e in surviving_events:
            etype = e.get("type", "EVENT").upper()
            context_lines.append(f"- [{etype}] {e['text']}")
        
        context_str = "\n".join(context_lines)
        
        chunk_kg = []
        for char, status in kg.get("statuses", {}).items():
            if char.lower() in chunk_text.lower():
                chunk_kg.append(f"- {char} is currently {status}.")
        
        enriched = f"{chunk_text}\n\n--- NARRATIVE STATE ---\nEvents:\n{context_str}"
        if chunk_kg:
            enriched += "\n\nCharacter Status:\n" + "\n".join(chunk_kg)
            
        dual_channel_memories.append(enriched)

    return dual_channel_memories


def build_config_C6(text: str, days_gap: int) -> list:
    """C6: + Semantic deduplication."""
    from src.segmenter import semantic_chunk
    from src.events import (
        extract_events, get_event_threshold,
        deduplicate_events_semantic,
    )
    from src.temporal.multi_scale_decay import apply_multi_scale_decay
    from src.knowledge.graph_builder import build_global_knowledge_graph

    chunks = semantic_chunk(text, threshold=0.75, min_sentences=3)
    chunk_events_map = []
    all_events = []
    offset = 0
    for chunk in chunks:
        events = extract_events(chunk)
        for e in events:
            e["position"] += offset
        chunk_events_map.append((chunk, events))
        all_events.extend(events)
        offset += max((e["position"] for e in events), default=0) + 1

    # Semantic dedup before decay
    all_events_dedup = deduplicate_events_semantic(all_events)

    kg = build_global_knowledge_graph(all_events_dedup, output_path="data/knowledge_ablation.json")
    apply_multi_scale_decay(all_events_dedup)

    # Filter on decay_score, keeping top 75% by decayed importance.
    scores = [e.get("decay_score", e["importance"]) for e in all_events_dedup]
    score_threshold = sorted(scores)[int(len(scores) * 0.25)] if scores else 0
    dedup_ids = {id(e) for e in all_events_dedup}

    dual_channel_memories = []
    for chunk_text, events in chunk_events_map:
        surviving_events = [
            e for e in events 
            if id(e) in dedup_ids and e.get("decay_score", e["importance"]) >= score_threshold
        ]
        
        if not surviving_events:
            continue
            
        context_lines = []
        for e in surviving_events:
            etype = e.get("type", "EVENT").upper()
            context_lines.append(f"- [{etype}] {e['text']}")
        
        context_str = "\n".join(context_lines)
        
        chunk_kg = []
        for char, status in kg.get("statuses", {}).items():
            if char.lower() in chunk_text.lower():
                chunk_kg.append(f"- {char} is currently {status}.")
        
        enriched = f"{chunk_text}\n\n--- NARRATIVE STATE ---\nEvents:\n{context_str}"
        if chunk_kg:
            enriched += "\n\nCharacter Status:\n" + "\n".join(chunk_kg)
            
        dual_channel_memories.append(enriched)

    return dual_channel_memories


def build_config_C7(text: str, days_gap: int) -> list:
    """C7: Full Chronos-RAG (all components — the production pipeline)."""
    # This mirrors main.py exactly
    from src.segmenter import semantic_chunk
    from src.events import (
        extract_events, apply_temporal_decay,
        get_event_threshold,
    )
    from src.knowledge.graph_builder import build_global_knowledge_graph

    chunks = semantic_chunk(text, threshold=0.75, min_sentences=3)
    chunk_events_map = []
    all_events = []
    offset = 0
    for chunk in chunks:
        events = extract_events(chunk)
        for e in events:
            e["position"] += offset
        chunk_events_map.append((chunk, events))
        all_events.extend(events)
        offset += max((e["position"] for e in events), default=0) + 1

    kg = build_global_knowledge_graph(all_events, output_path="data/knowledge_ablation.json")
    apply_temporal_decay(all_events)

    # Filter on decay_score, keeping top 75% by decayed importance.
    scores = [e.get("decay_score", e["importance"]) for e in all_events]
    score_threshold = sorted(scores)[int(len(scores) * 0.25)] if scores else 0

    dual_channel_memories = []
    for chunk_text, events in chunk_events_map:
        surviving_events = [e for e in events if e.get("decay_score", e["importance"]) >= score_threshold]
        
        if not surviving_events:
            continue
            
        context_lines = []
        for e in surviving_events:
            etype = e.get("type", "EVENT").upper()
            context_lines.append(f"- [{etype}] {e['text']}")
        
        context_str = "\n".join(context_lines)
        
        chunk_kg = []
        for char, status in kg.get("statuses", {}).items():
            if char.lower() in chunk_text.lower():
                chunk_kg.append(f"- {char} is currently {status}.")
        
        enriched = f"{chunk_text}\n\n--- NARRATIVE STATE ---\nEvents:\n{context_str}"
        if chunk_kg:
            enriched += "\n\nCharacter Status:\n" + "\n".join(chunk_kg)
            
        dual_channel_memories.append(enriched)

    return dual_channel_memories


CONFIGS = {
    "C0": ("Vanilla RAG",                        build_config_C0, False),
    "C1": ("+ Semantic Chunking",                 build_config_C1, False),
    "C2": ("+ Event Classification",              build_config_C2, False),
    "C3": ("+ Single-Scale Decay (hand-tuned)",   build_config_C3, False),
    "C4": ("+ Multi-Scale Decay (Atkinson-Shiffrin)", build_config_C4, False),
    "C5": ("+ Knowledge Graph (G-RAG)",           build_config_C5, False),
    "C6": ("+ Semantic Deduplication",             build_config_C6, False),
    "C7": ("Full Chronos-RAG",                     build_config_C7, False),
    "C8": ("+ Hybrid Search (BM25 + RRF)",         build_config_C7, True),
}


# ── Evaluation runner ─────────────────────────────────────────────────────────

def evaluate_config(
    config_id: str,
    text: str,
    qa_pairs: list,
    days_gap: int,
) -> dict:
    """Runs a single ablation configuration and evaluates it."""
    name, builder, use_hybrid = CONFIGS[config_id]
    print(f"\n{'='*60}")
    print(f"  Evaluating: {config_id} — {name}")
    print(f"{'='*60}")

    from src.embeddings import embed_memories, embed_text
    from src.vector_store import create_chroma_collection

    # Build memories
    t0 = time.time()
    try:
        memories = builder(text, days_gap)
    except Exception as e:
        print(f"  ✗ Pipeline failed: {e}")
        return {
            "config_id": config_id,
            "config_name": name,
            "error": str(e),
        }
    build_time = time.time() - t0

    if not memories:
        return {
            "config_id": config_id,
            "config_name": name,
            "memory_count": 0,
            "error": "No memories produced",
        }

    print(f"  Memories: {len(memories)}")
    print(f"  Build time: {build_time:.1f}s")

    # Embed and store
    collection_name = f"ablation_{config_id}"
    try:
        embeddings = embed_memories(memories)
        client, collection = create_chroma_collection(
            collection_name=collection_name,
            persist_dir="chroma_store"
        )
        # Clear and recreate
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        collection.add(
            embeddings=embeddings,
            documents=memories,
            ids=[f"m_{i}" for i in range(len(memories))],
        )
    except Exception as e:
        print(f"  ✗ Embedding/storage failed: {e}")
        return {
            "config_id": config_id,
            "config_name": name,
            "memory_count": len(memories),
            "error": str(e),
        }

    from src.hybrid_retrieval import hybrid_search

    # Evaluate on QA pairs
    recalls = []
    precisions = []
    sem_recalls = []

    for qa in qa_pairs:
        try:
            q_vec = embed_text(qa["question"])
            if use_hybrid:
                contexts = hybrid_search(
                    query=qa["question"], 
                    collection=collection, 
                    documents=memories, 
                    embed_fn=embed_text, 
                    k=min(5, len(memories))
                )
            else:
                results = collection.query(
                    query_embeddings=[q_vec],
                    n_results=min(5, len(memories)),
                    include=["documents"]
                )
                contexts = results["documents"][0] if results["documents"] else []
                
            r = keyword_recall(contexts, qa["ground_truth"])
            p = precision_at_k(contexts, qa["ground_truth"])
            sr = semantic_recall(contexts, qa["ground_truth"], embed_text)
            recalls.append(r)
            precisions.append(p)
            sem_recalls.append(sr)
        except Exception as e:
            print(f"  ✗ Eval error on QA {qa['id']}: {e}")
            recalls.append(0.0)
            precisions.append(0.0)
            sem_recalls.append(0.0)

    # Cleanup
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    mean_recall = sum(recalls) / len(recalls) if recalls else 0
    mean_precision = sum(precisions) / len(precisions) if precisions else 0
    mean_sem_recall = sum(sem_recalls) / len(sem_recalls) if sem_recalls else 0

    result = {
        "config_id": config_id,
        "config_name": name,
        "memory_count": len(memories),
        "build_time_sec": round(build_time, 2),
        "keyword_recall": round(mean_recall, 4),
        "precision_at_5": round(mean_precision, 4),
        "semantic_recall": round(mean_sem_recall, 4),
        "per_question": [
            {
                "id": qa["id"],
                "question_type": qa.get("question_type", "unknown"),
                "recall": round(r, 4),
                "precision": round(p, 4),
                "semantic_recall": round(sr, 4),
            }
            for qa, r, p, sr in zip(qa_pairs, recalls, precisions, sem_recalls)
        ],
    }

    print(f"  Keyword Recall:  {mean_recall:.4f}")
    print(f"  Semantic Recall: {mean_sem_recall:.4f}")
    print(f"  Precision@5:     {mean_precision:.4f}")

    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_ablation_results(results: list, output_dir: str):
    """Creates a grouped bar chart comparing all configurations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Ablation] matplotlib not installed — skipping plots")
        return

    valid = [r for r in results if "error" not in r]
    if not valid:
        return

    labels = [f"{r['config_id']}\n{r['config_name']}" for r in valid]
    recalls = [r["keyword_recall"] for r in valid]
    precisions = [r["precision_at_5"] for r in valid]

    x = range(len(valid))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar([i - width/2 for i in x], recalls, width, label="Keyword Recall",
                   color="#6366f1", alpha=0.85)
    bars2 = ax.bar([i + width/2 for i in x], precisions, width, label="Precision@5",
                   color="#22c55e", alpha=0.85)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Component-Level Impact on Retrieval Quality")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7, ha='center')
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0, min(1.1, max(recalls + precisions) + 0.15))

    plt.tight_layout()
    path = os.path.join(output_dir, "ablation_comparison.png")
    plt.savefig(path, dpi=150)
    print(f"\n[Ablation] Chart saved → {path}")
    plt.close()

    # ── Per-question-type breakdown ──────────────────────────────────────────
    # Compare C0 (vanilla) vs C8 (full) across question types
    c0 = next((r for r in valid if r["config_id"] == "C0"), None)
    best_id = "C8" if any(r["config_id"] == "C8" for r in valid) else "C7"
    c_best = next((r for r in valid if r["config_id"] == best_id), None)

    if c0 and c_best and "per_question" in c0 and "per_question" in c_best:
        qtypes = set(q["question_type"] for q in c0["per_question"])
        fig, ax = plt.subplots(figsize=(10, 5))

        type_data = {}
        for qt in sorted(qtypes):
            c0_scores = [q["recall"] for q in c0["per_question"] if q["question_type"] == qt]
            cbest_scores = [q["recall"] for q in c_best["per_question"] if q["question_type"] == qt]
            if c0_scores and cbest_scores:
                type_data[qt] = (
                    sum(c0_scores) / len(c0_scores),
                    sum(cbest_scores) / len(cbest_scores),
                )

        if type_data:
            labels = list(type_data.keys())
            c0_vals = [type_data[l][0] for l in labels]
            cbest_vals = [type_data[l][1] for l in labels]
            x = range(len(labels))

            ax.bar([i - 0.2 for i in x], c0_vals, 0.35, label="C0: Vanilla RAG",
                   color="#ef4444", alpha=0.7)
            ax.bar([i + 0.2 for i in x], cbest_vals, 0.35, label=f"{best_id}: Full Pipeline",
                   color="#6366f1", alpha=0.85)

            ax.set_ylabel("Keyword Recall")
            ax.set_title("Vanilla RAG vs Chronos-RAG by Question Type")
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels)
            ax.legend()
            ax.grid(True, alpha=0.2, axis="y")

            plt.tight_layout()
            path = os.path.join(output_dir, "ablation_by_question_type.png")
            plt.savefig(path, dpi=150)
            print(f"[Ablation] Question-type breakdown → {path}")
            plt.close()


def generate_markdown_table(results: list, output_dir: str):
    """Generates a markdown table for papers and README."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        return

    lines = [
        "# Ablation Study Results",
        "",
        "| Config | Description | Memories | Build Time | Keyword Recall | Precision@5 | Δ Recall |",
        "|--------|-------------|----------|------------|----------------|-------------|----------|",
    ]

    baseline_recall = valid[0]["keyword_recall"] if valid else 0

    for r in valid:
        delta = r["keyword_recall"] - baseline_recall
        delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
        lines.append(
            f"| {r['config_id']} | {r['config_name']} | "
            f"{r['memory_count']} | {r['build_time_sec']:.1f}s | "
            f"**{r['keyword_recall']:.4f}** | {r['precision_at_5']:.4f} | "
            f"{delta_str} |"
        )

    lines.append("")

    # Add improvement summary
    if len(valid) >= 2:
        c0_recall = valid[0]["keyword_recall"]
        best = max(valid, key=lambda r: r["keyword_recall"])
        if c0_recall > 0:
            improvement = ((best["keyword_recall"] - c0_recall) / c0_recall) * 100
            lines.extend([
                f"**Best configuration:** {best['config_id']} ({best['config_name']})",
                f"**Improvement over Vanilla RAG:** {improvement:+.1f}% keyword recall",
                "",
            ])

    path = os.path.join(output_dir, "ablation_table.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[Ablation] Markdown table → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study comparing Chronos-RAG components"
    )
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                        help="Configuration IDs to evaluate (default: all)")
    parser.add_argument("--days-gap", type=int, default=7,
                        help="Simulated days since last read (default: 7)")
    parser.add_argument("--qa", default=QA_PATH,
                        help="Path to QA pairs JSON")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--chapter-filter", default=None,
                        help="Only use QA pairs from this chapter (e.g., PROLOGUE)")
    args = parser.parse_args()

    # Load data
    print("[Ablation] Loading data...")
    text = get_text()

    with open(args.qa, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    # Filter QA pairs by chapter if requested
    if args.chapter_filter:
        qa_pairs = [qa for qa in qa_pairs if qa.get("chapter", "").upper() == args.chapter_filter.upper()]
        print(f"[Ablation] Filtered to {len(qa_pairs)} QA pairs from {args.chapter_filter}")

    print(f"[Ablation] Text: {len(text):,} chars")
    print(f"[Ablation] QA pairs: {len(qa_pairs)}")
    print(f"[Ablation] Days gap: {args.days_gap}")
    print(f"[Ablation] Configs: {', '.join(args.configs)}")

    # Run evaluations
    results = []
    for config_id in args.configs:
        if config_id not in CONFIGS:
            print(f"[Ablation] Unknown config: {config_id}, skipping")
            continue
        result = evaluate_config(config_id, text, qa_pairs, args.days_gap)
        results.append(result)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Ablation] Results saved → {results_path}")

    # Generate outputs
    plot_ablation_results(results, args.output_dir)
    generate_markdown_table(results, args.output_dir)

    # Print summary
    valid = [r for r in results if "error" not in r]
    if valid:
        print("\n" + "=" * 80)
        print("  ABLATION STUDY SUMMARY")
        print("=" * 80)
        print(f"  {'Config':<5} {'Description':<40} {'KW Recall':>10} {'Sem Recall':>10} {'P@5':>8}")
        print(f"  {'-'*75}")
        for r in valid:
            print(f"  {r['config_id']:<5} {r['config_name']:<40} "
                  f"{r['keyword_recall']:>10.4f} {r.get('semantic_recall', 0):>10.4f} "
                  f"{r['precision_at_5']:>8.4f}")

        best = max(valid, key=lambda r: r.get("semantic_recall", 0))
        print(f"\n  Best: {best['config_id']} ({best['config_name']}) "
              f"— {best.get('semantic_recall', 0):.4f} semantic recall")


if __name__ == "__main__":
    main()
