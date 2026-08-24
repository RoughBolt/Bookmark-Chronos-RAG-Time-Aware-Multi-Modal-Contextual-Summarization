import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

print("[Debug] Imports starting...")
import argparse
import json
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.ablation_study import build_config_C7
from experiments.baseline_comparison import chunk_naive
from src.embeddings import embed_memories, embed_text
from src.hybrid_retrieval import hybrid_search
print("[Debug] Importing evaluate_generation...")
from experiments.evaluate_generation import generate_answer, judge_answer
import chromadb

try:
    print("[Debug] Importing datasets...")
    from datasets import load_dataset
except ImportError:
    print("[Error] 'datasets' library not installed. Run: pip install datasets")
    sys.exit(1)

def run_pipeline(text, question, ground_truth, model, client, collection_name, is_naive):
    if is_naive:
        memories = chunk_naive(text)
    else:
        memories = build_config_C7(text, days_gap=7)

    if not memories:
        return 0, 0

    embeddings = embed_memories(memories)
    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    
    collection.add(
        embeddings=embeddings,
        documents=memories,
        ids=[f"m_{j}" for j in range(len(memories))]
    )
    
    contexts = hybrid_search(query=question, collection=collection, documents=memories, embed_fn=embed_text, k=5)
    
    if not contexts:
        return 0, 0
        
    gen_answer = generate_answer(question, contexts, model=model)
    scores = judge_answer(question, ground_truth, gen_answer, contexts, model=model)
    return scores["relevance"], scores["faithfulness"]

def main():
    parser = argparse.ArgumentParser(description="Evaluate on NarrativeQA subset (Chronos vs Vanilla)")
    parser.add_argument("--samples", type=int, default=3, help="Number of documents to sample")
    parser.add_argument("--model", type=str, default="llama3", help="Ollama model to use")
    args = parser.parse_args()

    print(f"\n[Eval] Loading NarrativeQA (small slice)...")
    dataset = load_dataset("deepmind/narrativeqa", split="test", streaming=True)
    iterator = iter(dataset)
    
    total_rel_c = 0
    total_faith_c = 0
    total_rel_n = 0
    total_faith_n = 0
    count = 0
    
    client = chromadb.Client()
    
    for i in range(args.samples):
        print(f"\n--- Document {i+1} ---")
        try:
            item = next(iterator)
        except StopIteration:
            break
            
        try:
            text = item["document"]["summary"]["text"]
            question = item["question"]["text"]
            ground_truth = item["answers"][0]["text"]
        except KeyError:
            print(f"Skipping index {i} due to structural mismatch")
            continue
        
        print(f"Q: {question}")
        
        # 1. Run Chronos-RAG
        print("[Debug] Running Chronos-RAG pipeline...")
        rel_c, faith_c = run_pipeline(text, question, ground_truth, args.model, client, f"nqa_c_{i}", is_naive=False)
        print(f"  Chronos-RAG -> Rel: {rel_c}/5 | Faith: {faith_c}/5")

        # 2. Run Vanilla RAG
        print("[Debug] Running Vanilla RAG pipeline...")
        rel_n, faith_n = run_pipeline(text, question, ground_truth, args.model, client, f"nqa_n_{i}", is_naive=True)
        print(f"  Vanilla RAG -> Rel: {rel_n}/5 | Faith: {faith_n}/5")
        
        total_rel_c += rel_c
        total_faith_c += faith_c
        total_rel_n += rel_n
        total_faith_n += faith_n
        count += 1

    if count > 0:
        print("\n" + "="*60)
        print(" NARRATIVEQA METRICS (Chronos-RAG vs Vanilla RAG)")
        print("="*60)
        print(" [CHRONOS-RAG]")
        print(f"   Average Relevance:    {total_rel_c/count:.2f} / 5.0")
        print(f"   Average Faithfulness: {total_faith_c/count:.2f} / 5.0")
        print("\n [VANILLA RAG]")
        print(f"   Average Relevance:    {total_rel_n/count:.2f} / 5.0")
        print(f"   Average Faithfulness: {total_faith_n/count:.2f} / 5.0")
        print("="*60)
        
        # Save results out
        with open("experiments/narrativeqa_results.json", "w") as f:
            json.dump({
                "samples": count,
                "model": args.model,
                "Chronos-RAG": {"relevance": total_rel_c/count, "faithfulness": total_faith_c/count},
                "Vanilla RAG": {"relevance": total_rel_n/count, "faithfulness": total_faith_n/count}
            }, f, indent=2)

if __name__ == "__main__":
    main()
