import os
import sys
import json
sys.path.insert(0, os.path.abspath("."))

from experiments.ablation_study import get_text, build_config_C7
from src.embeddings import embed_memories, embed_text
from src.hybrid_retrieval import hybrid_search
from src.vector_store import create_chroma_collection

print("Loading data...")
text = get_text()

with open("benchmark/qa_pairs.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

memories = build_config_C7(text, days_gap=7)

embeddings = embed_memories(memories)
collection_name = "eval_generation_debug"
client, collection = create_chroma_collection(collection_name, persist_dir="chroma_store")
try:
    client.delete_collection(collection_name)
except:
    pass
collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
collection.add(
    embeddings=embeddings,
    documents=memories,
    ids=[f"m_{i}" for i in range(len(memories))]
)

# Test E08
qa = next(q for q in qa_pairs if q["id"] == "E08")
print("\nQuestion:", qa["question"])

contexts = hybrid_search(
    query=qa["question"], 
    collection=collection, 
    documents=memories, 
    embed_fn=embed_text, 
    k=3
)

print("\nRetrieved Contexts:")
for i, c in enumerate(contexts):
    print(f"\n--- Context {i+1} ---\n{c}")

try:
    client.delete_collection(collection_name)
except:
    pass
