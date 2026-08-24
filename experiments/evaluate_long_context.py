import os
import json
import argparse
from dotenv import load_dotenv

# Load API keys securely from ~/.env
load_dotenv(os.path.expanduser("~/.env"))

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("[Error] Please install google-genai: pip install google-genai")
    sys.exit(1)

from experiments.evaluate_generation import judge_answer
from src.preprocessing import clean_text
from src.utils import find_chapters, extract_text_upto_chapter
from src.bookmark import load_bookmark

BOOK_PATH = "data/book.txt"
QA_PATH = "benchmark/qa_pairs.json"

def evaluate_local_dataset(client, model_name):
    print(f"\n[Eval] Loading Local Dataset (Book)")
    bookmark = load_bookmark()
    chapters = find_chapters(BOOK_PATH)
    text = extract_text_upto_chapter(BOOK_PATH, chapters, bookmark["pov"], bookmark["occurrence"])
    text = clean_text(text)

    with open(QA_PATH, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    # Use only Prologue QA pairs for consistency with baseline_comparison
    qa_eval = [qa for qa in qa_pairs if qa.get("chapter", "") == "PROLOGUE"]
    if not qa_eval:
        qa_eval = qa_pairs[:20]

    total_rel = 0
    total_faith = 0
    count = 0

    print(f"[Eval] Evaluating {model_name} on {len(qa_eval)} QA pairs...")
    
    for qa in qa_eval:
        question = qa["question"]
        ground_truth = qa["ground_truth"]
        
        prompt = f"""You are answering questions based on the following complete text.
Do not use outside knowledge. Answer accurately.

Text:
{text}

Question: {question}"""

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            answer = response.text
            
            # Use Ollama to judge the answer for consistency with other evaluations
            # We pass the full text as context for faithfulness evaluation
            scores = judge_answer(question, ground_truth, answer, [text], model="qwen2:0.5b")
            total_rel += scores["relevance"]
            total_faith += scores["faithfulness"]
            count += 1
            print(f"  Q: {question}\n  -> Rel: {scores['relevance']}/5 | Faith: {scores['faithfulness']}/5")
        except Exception as e:
            print(f"  [Error] Failed to generate answer: {e}")

    if count > 0:
        print(f"\n[Result] Local Dataset Average Relevance: {total_rel/count:.2f} / 5.0")
        print(f"[Result] Local Dataset Average Faithfulness: {total_faith/count:.2f} / 5.0")
        return {"relevance": total_rel/count, "faithfulness": total_faith/count, "samples": count}
    return None

def evaluate_narrativeqa(client, model_name, samples=5):
    print(f"\n[Eval] Loading NarrativeQA Dataset")
    from datasets import load_dataset
    dataset = load_dataset("deepmind/narrativeqa", split="test", streaming=True)
    iterator = iter(dataset)
    
    total_rel = 0
    total_faith = 0
    count = 0
    
    for i in range(samples):
        try:
            item = next(iterator)
        except StopIteration:
            break
            
        try:
            text = item["document"]["summary"]["text"]
            question = item["question"]["text"]
            ground_truth = item["answers"][0]["text"]
        except KeyError:
            continue
            
        prompt = f"""You are answering questions based on the following complete text.
Do not use outside knowledge. Answer accurately.

Text:
{text}

Question: {question}"""

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            answer = response.text
            
            scores = judge_answer(question, ground_truth, answer, [text], model="qwen2:0.5b")
            total_rel += scores["relevance"]
            total_faith += scores["faithfulness"]
            count += 1
            print(f"  Q: {question}\n  -> Rel: {scores['relevance']}/5 | Faith: {scores['faithfulness']}/5")
        except Exception as e:
            print(f"  [Error] Failed to generate answer: {e}")

    if count > 0:
        print(f"\n[Result] NarrativeQA Average Relevance: {total_rel/count:.2f} / 5.0")
        print(f"[Result] NarrativeQA Average Faithfulness: {total_faith/count:.2f} / 5.0")
        return {"relevance": total_rel/count, "faithfulness": total_faith/count, "samples": count}
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate Long-Context Models")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro", help="Gemini model name")
    parser.add_argument("--samples", type=int, default=5, help="NarrativeQA samples")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[Error] GEMINI_API_KEY not found in environment or ~/.env")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    
    print("=" * 60)
    print(f" Evaluating Long-Context Model: {args.model}")
    print("=" * 60)
    
    local_res = evaluate_local_dataset(client, args.model)
    nqa_res = evaluate_narrativeqa(client, args.model, args.samples)
    
    with open("experiments/long_context_results.json", "w") as f:
        json.dump({
            "model": args.model,
            "local": local_res,
            "narrativeqa": nqa_res
        }, f, indent=2)
        
    print("\n[Done] Results saved to long_context_results.json")

if __name__ == "__main__":
    main()
