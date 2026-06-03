import os
# Disable tensorflow to prevent grpc lock blocking deadlocks on macOS
os.environ["USE_TF"] = "0"
# Also disable tokenizers parallelism just in case
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.utils import (
    find_chapters,
    extract_text_upto_chapter,
    days_since_last_read,
    get_summary_level
)
from src.events import (
    extract_events,
    apply_temporal_decay,
    get_event_threshold,
    get_event_limit
)
from src.bookmark import load_bookmark
from src.summarizer import summarize_text
from src.preprocessing import clean_text
from src.segmenter import semantic_chunk
from src.event_chain import build_narrative
from src.scene_build import build_scenes
from src.memory_orchestrator import generate_memory_recall
from src.narrative_stitch import test_stitchers
from src.embeddings import embed_memories, embed_text, active_backend
from src.vector_store import create_chroma_collection, store_memories, clear_collection
from src.recall import recall_memories
from src.cache import cached_embed, cache_stats
from src.indexer import load_manifest, save_manifest, needs_reindex
from src.graph_recall import graph_augment
from src.uncertainty import compute_retrieval_uncertainty, is_uncertain
from src.profiler import Profiler
from src.knowledge.graph_builder import build_global_knowledge_graph

# ── Setup ─────────────────────────────────────────────────────────────────────

BOOK_PATH = "data/book.txt"

profiler = Profiler(enabled=True)
backend  = active_backend()

chapters = find_chapters(BOOK_PATH)
bookmark = load_bookmark()

days_gap = days_since_last_read(bookmark["last_read"])
level    = get_summary_level(days_gap)

print("\n=======================================================")
print(f"   🧠 MEMORY CORE ONLINE: {days_gap} DAYS SINCE LAST READ")
print(f"   📡 Embedding backend : {backend}")
print("=======================================================\n")

# ── Manifest check — skip re-indexing if bookmark hasn't moved ────────────────

manifest = load_manifest()
should_reindex, reindex_reason = needs_reindex(bookmark, manifest, current_backend=backend)

client, collection = create_chroma_collection()

if not should_reindex:
    print(f"[✓] Loaded existing index ({manifest['memory_count']} memories) — skipping re-embed.\n")
    memory = []   # Not needed for query loop; memories live in ChromaDB
else:
    print(f"[~] Reindexing ({reindex_reason})...\n")

    # ── Clear stale collection before full re-index ───────────────────────────
    collection = clear_collection(client)

    # ── Text extraction ───────────────────────────────────────────────────────
    text = extract_text_upto_chapter(
        BOOK_PATH,
        chapters,
        bookmark["pov"],
        bookmark["occurrence"]
    )
    text = clean_text(text)

    # ── Summarize (for human-readable recap) ──────────────────────────────────
    summary = summarize_text(text, level)
    print("===== STORY RECAP =====")
    print(summary)
    print("=======================\n")

    # ── Semantic chunking ─────────────────────────────────────────────────────
    with profiler.measure("semantic_chunking"):
        chunks = semantic_chunk(text, threshold=0.75, min_sentences=3)

    print(f"[Chunker] {len(chunks)} semantic chunks produced.\n")

    # ── Event extraction across all chunks (position-offset aware) ───────────
    with profiler.measure("event_extraction"):
        all_events = []
        position_offset = 0
        for chunk in chunks:
            chunk_events = extract_events(chunk)
            # Apply global position offset so events are globally ordered
            for e in chunk_events:
                e["position"] += position_offset
            all_events.extend(chunk_events)
            position_offset += max((e["position"] for e in chunk_events), default=0) + 1

    events = all_events

    # ── Knowledge graph (offline factual state tracking) ─────────────────────
    knowledge_graph = build_global_knowledge_graph(events)
    print("===== FACTUAL KNOWLEDGE GRAPH (OFFLINE) =====")
    print(f"Characters Tracked: {len(knowledge_graph['statuses'])}")
    for char, status in knowledge_graph["statuses"].items():
        print(f"  - {char}: {status}")
    print()

    # ── Exponential temporal decay: S = S_semantic · e^(−λ · Δt) ─────────────
    with profiler.measure("temporal_decay"):
        events = apply_temporal_decay(events)

    # ── Filter by days_gap threshold, sort chronologically ───────────────────
    threshold = get_event_threshold(days_gap)
    filtered_events = [e for e in events if e["importance"] >= threshold]
    filtered_events = sorted(filtered_events, key=lambda x: x["position"])

    # ── Narrative chaining → scene building → memory orchestration ───────────
    narrative_events = build_narrative(filtered_events)
    scenes           = build_scenes(narrative_events)
    memory           = generate_memory_recall(scenes)
    memory           = [m for m in memory if m.strip()]

    # ── Embed + store in ChromaDB ─────────────────────────────────────────────
    with profiler.measure("embed_memories"):
        embeddings = embed_memories(memory)

    with profiler.measure("vector_store"):
        store_memories(
            collection=collection,
            memories=memory,
            embeddings=embeddings
        )

    # ── Save manifest ─────────────────────────────────────────────────────────
    save_manifest(bookmark, memory_count=len(memory), embedding_backend=backend)
    print(f"[✓] Indexed {len(memory)} memories and saved manifest.\n")

    profiler.print_summary()

# ── Interactive query loop ─────────────────────────────────────────────────────

print(f"[Cache] {cache_stats()['entries']} cached queries loaded.")
print("[Search] Ready. Type your query or 'quit' to exit.\n")

while True:
    try:
        query = input("\n[Search] What do you want to recall? (or 'quit'): ")

        if query.strip().lower() in ["quit", "exit", "q"]:
            print("\nEntering Standby... Goodbye!\n")
            break

        if not query.strip():
            continue

        # ── Cached embedding ──────────────────────────────────────────────────
        with profiler.measure("embed_query"):
            query_vector = cached_embed(query, embed_text)

        # ── Uncertainty check — abstain if retrieval is unstable ──────────────
        with profiler.measure("uncertainty_check"):
            uncertainty = compute_retrieval_uncertainty(
                query_vector, collection, n_passes=5, sigma=0.01, k=5
            )

        if is_uncertain(uncertainty):
            print(f"\n[System] Retrieval uncertainty too high ({uncertainty:.2f}). "
                  f"Insufficient context to reconstruct a reliable narrative.")
            continue

        # ── Vector recall (time-aware, distance back-off) ─────────────────────
        with profiler.measure("vector_retrieval"):
            results = recall_memories(
                collection,
                query_embedding=query_vector,
                days_gap=days_gap
            )

        if not results:
            print("\n[System] Memory threshold too strict. No exact matches found for that query.")
            continue

        # ── G-RAG: graph neighborhood expansion ──────────────────────────────
        with profiler.measure("graph_augment"):
            results = graph_augment(results, collection)

        # ── Hybrid factual preamble (for gaps > 3 days) ───────────────────────
        preamble = ""
        if days_gap > 3:
            from src.hybrid_recall import generate_hybrid_context
            preamble = generate_hybrid_context(results) + "\n\n"

        # ── Narrative stitching (3 algorithms) ───────────────────────────────
        with profiler.measure("narrative_stitch"):
            stitched_results = test_stitchers(results)

        # ── Output ────────────────────────────────────────────────────────────
        print(f"\n===== RECONSTRUCTED NARRATIVE ({len(results)} Fragments Fetched) =====")
        print(f"[Uncertainty: {uncertainty:.2f}  |  G-RAG expanded: {len(results)} fragments]")
        print("\n--- ALGORITHM A (Template/Rule-Based Matrix) ---")
        print(preamble + stitched_results["A_Template"])
        print("\n--- ALGORITHM B (Syntactic Fusion / SVO Merging) ---")
        print(preamble + stitched_results["B_Syntax"])
        print("\n--- ALGORITHM C (Lexical / Metadata Pacing) ---")
        print(preamble + stitched_results["C_Pacing"])
        print("\n===============================================================")

        profiler.print_summary()

    except KeyboardInterrupt:
        print("\nEntering Standby... Goodbye!\n")
        break
