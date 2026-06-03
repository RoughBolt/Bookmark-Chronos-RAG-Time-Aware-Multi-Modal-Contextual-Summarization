"""
RAGAS Benchmark: Baseline RAG vs Chronos-RAG
----------------------------------------------
Evaluates retrieval quality using the RAGAS framework.

Metrics:
  - context_precision   : Are retrieved chunks actually relevant?
  - context_recall      : Does retrieval cover what's needed to answer?
  - faithfulness        : Is the generated answer grounded in retrieved context?
  - answer_relevancy    : Does the answer actually address the question?

Setup:
  pip install ragas datasets sentence-transformers

Usage:
  python3 -m benchmark.run_ragas

Note: RAGAS uses an LLM judge. This script defaults to Ollama (llama3).
      For GPT-4 judging, set OPENAI_API_KEY in your environment.
"""

import sys
import os
import json

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embeddings import embed_text, embed_memories, active_backend
from src.vector_store import create_chroma_collection, store_memories, clear_collection
from src.recall import recall_memories
from src.narrative_stitch import test_stitchers
from src.utils import find_chapters, extract_text_upto_chapter, days_since_last_read, get_summary_level
from src.bookmark import load_bookmark
from src.preprocessing import clean_text
from src.events import extract_events, apply_temporal_decay, get_event_threshold
from src.event_chain import build_narrative
from src.scene_build import build_scenes
from src.memory_orchestrator import generate_memory_recall
from src.graph_recall import graph_augment


BOOK_PATH = "data/book.txt"
QA_PATH = os.path.join(os.path.dirname(__file__), "qa_pairs.json")


# ── Build baseline collection (vanilla RAG — no decay, no graph) ──────────────

def build_baseline_collection():
    """Standard RAG: embed sentences directly, no temporal scoring."""
    print("\n[Benchmark] Building baseline RAG index...")
    bookmark = load_bookmark()
    chapters = find_chapters(BOOK_PATH)

    text = extract_text_upto_chapter(BOOK_PATH, chapters, bookmark["pov"], bookmark["occurrence"])
    text = clean_text(text)

    # Simple sentence-level splitting — no semantic chunking
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]

    client, collection = create_chroma_collection(
        collection_name="benchmark_baseline",
        persist_dir="chroma_store"
    )
    clear_collection(client, "benchmark_baseline")
    collection = client.create_collection(
        name="benchmark_baseline",
        metadata={"hnsw:space": "cosine", "hnsw:ef": 50, "hnsw:M": 16}
    )

    print(f"[Benchmark] Embedding {len(sentences)} sentences (baseline)...")
    embeddings = embed_memories(sentences)
    store_memories(collection, sentences, embeddings)

    print(f"[Benchmark] Baseline index ready ({len(sentences)} entries)")
    return collection


def build_chronos_collection():
    """Chronos-RAG: full pipeline with temporal decay, semantic chunking, G-RAG."""
    print("\n[Benchmark] Building Chronos-RAG index...")
    bookmark = load_bookmark()
    chapters = find_chapters(BOOK_PATH)
    days_gap = days_since_last_read(bookmark["last_read"])

    text = extract_text_upto_chapter(BOOK_PATH, chapters, bookmark["pov"], bookmark["occurrence"])
    text = clean_text(text)

    events = extract_events(text)
    events = apply_temporal_decay(events)

    threshold = get_event_threshold(days_gap)
    filtered_events = [e for e in events if e["importance"] >= threshold]
    filtered_events = sorted(filtered_events, key=lambda x: x["position"])

    narrative_events = build_narrative(filtered_events)
    scenes = build_scenes(narrative_events)
    memory = generate_memory_recall(scenes)
    memory = [m for m in memory if m.strip()]

    client, collection = create_chroma_collection(
        collection_name="benchmark_chronos",
        persist_dir="chroma_store"
    )
    clear_collection(client, "benchmark_chronos")
    collection = client.create_collection(
        name="benchmark_chronos",
        metadata={"hnsw:space": "cosine", "hnsw:ef": 50, "hnsw:M": 16}
    )

    embeddings = embed_memories(memory)
    store_memories(collection, memory, embeddings)

    print(f"[Benchmark] Chronos-RAG index ready ({len(memory)} memories)")
    return collection, days_gap


# ── Retrieve contexts for a question ─────────────────────────────────────────

def retrieve_baseline(question: str, collection) -> list[str]:
    qvec = embed_text(question)
    results = collection.query(
        query_embeddings=[qvec],
        n_results=5,
        include=["documents"]
    )
    return results["documents"][0]


def retrieve_chronos(question: str, collection, days_gap: int) -> list[str]:
    qvec = embed_text(question)
    recalled = recall_memories(collection, qvec, days_gap)
    recalled = graph_augment(recalled, collection)
    return [r["text"] for r in recalled]


# ── Simple local answer generation via Ollama ─────────────────────────────────

def generate_answer(question: str, contexts: list[str]) -> str:
    """Uses Ollama llama3 to generate an answer from the retrieved contexts."""
    try:
        import ollama
        context_str = "\n\n".join(contexts)
        prompt = (
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            f"Answer the question using ONLY the context above. "
            f"If the context is insufficient, say 'Insufficient context'."
        )
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        # Fallback: use context directly as the "answer"
        return " ".join(contexts[:2]) if contexts else "No context retrieved."


# ── RAGAS evaluation ──────────────────────────────────────────────────────────

def run_ragas_eval(qa_pairs, answers_baseline, answers_chronos,
                   contexts_baseline, contexts_chronos):
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset

        def make_dataset(answers, contexts):
            return Dataset.from_dict({
                "question":     [q["question"] for q in qa_pairs],
                "answer":       answers,
                "contexts":     contexts,
                "ground_truth": [q["ground_truth"] for q in qa_pairs],
            })

        print("\n[RAGAS] Evaluating Baseline RAG...")
        ds_baseline = make_dataset(answers_baseline, contexts_baseline)
        result_baseline = evaluate(ds_baseline, metrics=[
            faithfulness, answer_relevancy, context_precision, context_recall
        ])

        print("[RAGAS] Evaluating Chronos-RAG...")
        ds_chronos = make_dataset(answers_chronos, contexts_chronos)
        result_chronos = evaluate(ds_chronos, metrics=[
            faithfulness, answer_relevancy, context_precision, context_recall
        ])

        return result_baseline, result_chronos

    except ImportError:
        print("\n[RAGAS] ragas/datasets not installed. Run: pip install ragas datasets")
        return None, None


# ── Results table printer ─────────────────────────────────────────────────────

def print_results_table(result_baseline, result_chronos):
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║           RAGAS Benchmark: Baseline vs Chronos-RAG       ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  {'Metric':<22} {'Baseline':>10} {'Chronos':>10} {'Δ':>8} ║")
    print("╠══════════════════════════════════════════════════════════╣")

    for m in metrics:
        b = result_baseline[m] if result_baseline else 0.0
        c = result_chronos[m] if result_chronos else 0.0
        delta = ((c - b) / b * 100) if b > 0 else 0.0
        sign = "+" if delta >= 0 else ""
        print(f"║  {m:<22} {b:>10.3f} {c:>10.3f} {sign}{delta:>6.1f}%  ║")

    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n[Benchmark] Embedding backend: {active_backend()}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open(QA_PATH, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    print(f"\n[Benchmark] Loaded {len(qa_pairs)} QA pairs from {QA_PATH}")
    print(f"[Benchmark] Embedding backend: {active_backend()}\n")

    # Build indices
    baseline_col = build_baseline_collection()
    chronos_col, days_gap = build_chronos_collection()

    # Retrieve + generate answers
    print("\n[Benchmark] Running retrieval + generation for all questions...")
    answers_baseline, answers_chronos = [], []
    contexts_baseline, contexts_chronos = [], []

    for i, qa in enumerate(qa_pairs):
        q = qa["question"]
        print(f"  [{i+1}/{len(qa_pairs)}] {q[:60]}...")

        ctx_b = retrieve_baseline(q, baseline_col)
        ctx_c = retrieve_chronos(q, chronos_col, days_gap)

        ans_b = generate_answer(q, ctx_b)
        ans_c = generate_answer(q, ctx_c)

        contexts_baseline.append(ctx_b)
        contexts_chronos.append(ctx_c)
        answers_baseline.append(ans_b)
        answers_chronos.append(ans_c)

    # RAGAS evaluation
    result_b, result_c = run_ragas_eval(
        qa_pairs, answers_baseline, answers_chronos,
        contexts_baseline, contexts_chronos
    )

    print_results_table(result_b, result_c)
