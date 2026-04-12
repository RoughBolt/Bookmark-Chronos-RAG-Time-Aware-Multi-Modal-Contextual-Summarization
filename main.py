from src.utils import (
    find_chapters,
    extract_text_upto_chapter,
    days_since_last_read,
    get_summary_level
)
from src.events import extract_events
from src.bookmark import load_bookmark
from src.summarizer import summarize_text
from src.preprocessing import clean_text
from src.segmenter import split_into_paragraphs
from src.hierarchical import (
    summarize_paragraphs,
    summarize_hierarchy
)
from src.events import (
    get_event_threshold,
    get_event_limit
)
from src.event_chain import build_narrative
from src.scene_build import build_scenes
from src.scene_abstraction import summarize_scene
from src.scene_abstraction import old_summarize_scene
from src.scene_consolidation import consolidate_scene
from src.memory_orchestrator import generate_memory_recall
from src.narrative_stitch import test_stitchers
from src.embeddings import embed_memories
from src.embeddings import embed_text
from src.vector_store import create_chroma_collection, store_memories
from src.recall import recall_memories

BOOK_PATH = "data/book.txt"

chapters = find_chapters(BOOK_PATH)
bookmark = load_bookmark()

days_gap = days_since_last_read(bookmark["last_read"])
level = get_summary_level(days_gap)

text = extract_text_upto_chapter(
    BOOK_PATH,
    chapters,
    bookmark["pov"],
    bookmark["occurrence"]
)

text = clean_text(text)

summary = summarize_text(text, level)                           # Generate summary based on level

# print("Days gap:", days_gap)
# print("Summary level:", level)
# print("\n===== SUMMARY =====\n")
# print(summary)

events = extract_events(text)                                   # Extract key events

# if level in ["medium", "long"]:
#     print("\n===== KEY EVENTS =====\n")
#     for e in events[:10]:
#         print("-", e)

# paragraphs = split_into_paragraphs(text)
# para_summaries = summarize_paragraphs(paragraphs)

# print("\n--- Paragraph Summaries (first 5) ---\n")
# for i, s in enumerate(para_summaries[:5], 1):
#     print(f"{i}. {s}")

# final_summary = summarize_hierarchy(para_summaries, level)

# print("\n===== HIERARCHICAL SUMMARY =====\n")
# print(final_summary)

# events = extract_events(text)

# UNMARK FROM HERE BELOW

bookmark_position = max(e["position"] for e in events)  # approx end of reading

for e in events:
    distance = abs(e["position"] - bookmark_position)
    e["decay_score"] = e["importance"] / (1 + distance)
    if e["type"] == "death":
        e["decay_score"] *= 2.5

threshold = get_event_threshold(days_gap)
limit = get_event_limit(days_gap)

filtered_events = [
    e for e in events if e["importance"] >= threshold
]

# Keep chronological order
filtered_events = sorted(filtered_events, key=lambda x: x["position"])

# 🔥 NEW: narrative chaining
narrative_events = build_narrative(filtered_events)

# print("\n--- Narrative Events ---\n")
# for e in narrative_events[:limit]:
#     print(f"[{e['type'].upper()}] {e['text']}")

scenes = build_scenes(narrative_events)

# print("\n--- VER 1 MEMORY SCENES ---\n")
# for s in scenes:
#     scene = old_summarize_scene(s)
#     print(f"[{scene['type'].upper()}] {scene['text']}")

# print("\n--- VER 2 MEMORY SCENES ---\n")
# for s in scenes:
#     scene = summarize_scene(s, days_gap)
#     print(f"[{scene['type'].upper()}] {scene['text']}")

memory = generate_memory_recall(scenes)
memory = [m for m in memory if m.strip()]

# print("MEMORY:", memory)

# print(scenes[0][0])

# print("\n===== RECALLABLE MEMORIES =====\n")

# stitched = stitch_memory_timeline(memory)
# print(stitched)

# print("\n\n\n\n\n\n\n\n\n")

tags = set(m.split("]")[0] + "]" for m in memory)
# print(tags)

embeddings = embed_memories(memory)

# print("Embeddings generated:")
# print("Count:", len(embeddings))
# print("Vector size:", len(embeddings[0]))

client, collection = create_chroma_collection()

store_memories(
    collection=collection,
    memories=memory,
    embeddings=embeddings
)

# print("✅ Memories stored in ChromaDB")

query = "important fight or death in the story"
query_vector = embed_text(query) 

test_gaps = [1, 7, 90]

for gap in test_gaps:
    print(f"\n===============================================")
    print(f"       TESTING TIME-AWARE RECALL: GAP = {gap} DAYS")
    print(f"===============================================\n")

    results = recall_memories(
        collection,
        query_embedding=query_vector,
        days_gap=gap
    )
    
    print(f"--- FILTERED RECALL RAW (Vectors = {len(results)}) ---")
    for r in results:
        print(f"[{r['tag']}] ({round(r['distance'],3)}) {r['text']}")

    stitched_results = test_stitchers(results)

    print("\n--- ALGORITHM A (Template/Rule-Based Matrix) ---")
    print(stitched_results["A_Template"])
    print("\n--- ALGORITHM B (Syntactic Fusion / SVO Merging) ---")
    print(stitched_results["B_Syntax"])
    print("\n--- ALGORITHM C (Lexical / Metadata Pacing) ---")
    print(stitched_results["C_Pacing"])
    print("\n")
