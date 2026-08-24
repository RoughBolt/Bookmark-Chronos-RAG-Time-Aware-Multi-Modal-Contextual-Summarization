"""
LLM-as-a-Judge Evaluation Pipeline
====================================
Evaluates the generative performance of Chronos-RAG using a local LLM.
For a sample of questions, it:
  1. Retrieves context using the best pipeline (C8: Hybrid Search + Sklearn Classifier)
  2. Generates an answer using Ollama (Llama 3)
  3. Judges the generated answer on Faithfulness (hallucination check) and Relevance (correctness)

Usage:
  python -m experiments.evaluate_generation --samples 30
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.ablation_study import get_text, build_config_C7
from src.embeddings import embed_memories, embed_text
from src.hybrid_retrieval import hybrid_search
from src.vector_store import create_chroma_collection

try:
    import ollama
except ImportError:
    print("[Error] ollama python package not installed. Run: pip install ollama")
    sys.exit(1)


def generate_answer(question: str, contexts: list[str], model: str = "llama3") -> str:
    """Generates an answer using the provided contexts."""
    context_str = "\n\n".join(contexts)
    prompt = f"""You are a helpful assistant. Answer the following question based ONLY on the provided context. If the context does not contain the answer, say "I don't know".

Context:
{context_str}

Question: {question}
Answer:"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1}
    )
    return response["message"]["content"].strip()


def judge_answer(question: str, ground_truth: str, generated: str, contexts: list[str], model: str = "llama3") -> dict:
    """
    Uses the LLM as a judge to evaluate the generated answer.
    Returns a dict with 'relevance' (1-5) and 'faithfulness' (1-5).
    """
    context_str = "\n\n".join(contexts)
    prompt = f"""You are an impartial judge evaluating an AI system.

Question: {question}
Ground Truth Answer: {ground_truth}
Generated Answer: {generated}
Retrieved Context used for generation:
{context_str}

Please evaluate the Generated Answer on two metrics from 1 to 5:
1. Relevance: How well does the generated answer match the factual correctness of the ground truth? (1 = completely wrong, 5 = perfectly captures the ground truth)
2. Faithfulness: Is the generated answer entirely supported by the Retrieved Context? (1 = hallucinated information not in context, 5 = perfectly faithful to context)

Respond ONLY with a JSON object in this exact format:
{{"relevance": <int>, "faithfulness": <int>}}
Do not include any other text."""

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "format": "json"}
        )
        content = response["message"]["content"].strip()
        scores = json.loads(content)
        return {
            "relevance": int(scores.get("relevance", 1)),
            "faithfulness": int(scores.get("faithfulness", 1))
        }
    except Exception as e:
        print(f"[Judge Error] {e}")
        return {"relevance": 1, "faithfulness": 1}


def main():
    parser = argparse.ArgumentParser(description="Evaluate generation using LLM-as-a-judge")
    parser.add_argument("--samples", type=int, default=10, help="Number of questions to sample")
    parser.add_argument("--model", type=str, default="llama3", help="Ollama model to use")
    parser.add_argument("--qa", default="benchmark/qa_pairs.json", help="Path to QA pairs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"\n[Eval] Loading data and preparing Chronos-RAG pipeline...")
    text = get_text()
    
    with open(args.qa, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    # Sample QA pairs
    sampled_qa = random.sample(qa_pairs, min(args.samples, len(qa_pairs)))
    
    # 1. Build Memories (C7 pipeline)
    print("[Eval] Segmenting text and extracting events (Full Pipeline)...")
    memories = build_config_C7(text, days_gap=7)
    
    # 2. Embed and Store
    print(f"[Eval] Embedding {len(memories)} memories...")
    collection_name = "eval_generation_c8"
    embeddings = embed_memories(memories)
    
    client, collection = create_chroma_collection(collection_name, persist_dir="chroma_store")
    try:
         client.delete_collection(collection_name)
    except Exception:
         pass
    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    collection.add(
         embeddings=embeddings,
         documents=memories,
         ids=[f"m_{i}" for i in range(len(memories))]
    )

    # 3. Evaluate Generation
    print(f"\n[Eval] Running LLM-as-a-Judge Evaluation on {len(sampled_qa)} samples...")
    print(f"       Using model: {args.model}")
    print("-" * 70)

    results = []
    total_relevance = 0
    total_faithfulness = 0

    for i, qa in enumerate(sampled_qa):
        q = qa["question"]
        gt = qa["ground_truth"]
        
        # Retrieve
        contexts = hybrid_search(
            query=q, 
            collection=collection, 
            documents=memories, 
            embed_fn=embed_text, 
            k=5
        )
        
        if not contexts:
            results.append({"id": qa["id"], "relevance": 1, "faithfulness": 1, "error": "No contexts retrieved"})
            continue

        # Generate
        gen_answer = generate_answer(q, contexts, model=args.model)
        
        # Judge
        scores = judge_answer(q, gt, gen_answer, contexts, model=args.model)
        
        rel = scores["relevance"]
        faith = scores["faithfulness"]
        
        total_relevance += rel
        total_faithfulness += faith
        
        results.append({
            "id": qa["id"],
            "question": q,
            "ground_truth": gt,
            "generated": gen_answer,
            "relevance": rel,
            "faithfulness": faith
        })
        
        print(f"[{i+1}/{len(sampled_qa)}] {qa['id']} | Rel: {rel}/5 | Faith: {faith}/5")

    # Cleanup
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    # Print Summary
    avg_rel = total_relevance / len(sampled_qa) if sampled_qa else 0
    avg_faith = total_faithfulness / len(sampled_qa) if sampled_qa else 0
    
    print("\n" + "="*50)
    print(" GENERATION METRICS (LLM-as-a-Judge)")
    print("="*50)
    print(f" Average Relevance:    {avg_rel:.2f} / 5.0")
    print(f" Average Faithfulness: {avg_faith:.2f} / 5.0")
    print("="*50)

    # Save
    os.makedirs("experiments", exist_ok=True)
    out_path = f"experiments/generation_results_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump({
            "summary": {
                "average_relevance": avg_rel,
                "average_faithfulness": avg_faith,
                "samples": len(sampled_qa)
            },
            "results": results
        }, f, indent=2)
    print(f"\n[Eval] Detailed results saved to {out_path}")


if __name__ == "__main__":
    main()
