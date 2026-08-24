import os
import json
import argparse
import subprocess
from dotenv import load_dotenv

# Load API keys securely from ~/.env
load_dotenv(os.path.expanduser("~/.env"))

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.evaluate_generation import judge_answer
from src.preprocessing import clean_text
from src.utils import find_chapters, extract_text_upto_chapter
from src.bookmark import load_bookmark

BOOK_PATH = "data/book.txt"
QA_PATH = "benchmark/qa_pairs.json"
GRAPHRAG_ROOT = "experiments/graphrag_workspace"

def setup_graphrag_workspace(text, workspace_dir):
    os.makedirs(os.path.join(workspace_dir, "input"), exist_ok=True)
    with open(os.path.join(workspace_dir, "input", "book.txt"), "w") as f:
        f.write(text)

    if not os.path.exists(os.path.join(workspace_dir, "settings.yaml")):
        subprocess.run([sys.executable, "-m", "graphrag.index", "--init", "--root", workspace_dir], check=True)
        
    print("[GraphRAG] Running indexing pipeline (this may take a while and cost OpenAI credits)...")
    # For a real benchmark, this would run to completion. We will simulate execution if it takes too long.
    subprocess.run([sys.executable, "-m", "graphrag.index", "--root", workspace_dir], check=True)


def evaluate_local_dataset(model_name):
    print(f"\n[Eval] Loading Local Dataset (Book)")
    bookmark = load_bookmark()
    chapters = find_chapters(BOOK_PATH)
    text = extract_text_upto_chapter(BOOK_PATH, chapters, bookmark["pov"], bookmark["occurrence"])
    text = clean_text(text)

    # Setup and index GraphRAG
    setup_graphrag_workspace(text, GRAPHRAG_ROOT)

    with open(QA_PATH, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    qa_eval = [qa for qa in qa_pairs if qa.get("chapter", "") == "PROLOGUE"]
    if not qa_eval:
        qa_eval = qa_pairs[:20]

    total_rel = 0
    total_faith = 0
    count = 0

    for qa in qa_eval:
        question = qa["question"]
        ground_truth = qa["ground_truth"]
        
        # Query GraphRAG globally
        result = subprocess.run(
            [sys.executable, "-m", "graphrag.query", "--root", GRAPHRAG_ROOT, "--method", "global", question],
            capture_output=True, text=True
        )
        answer = result.stdout
        
        scores = judge_answer(question, ground_truth, answer, [text], model="qwen2:0.5b")
        total_rel += scores["relevance"]
        total_faith += scores["faithfulness"]
        count += 1
        print(f"  Q: {question}\n  -> Rel: {scores['relevance']}/5 | Faith: {scores['faithfulness']}/5")

    if count > 0:
        return {"relevance": total_rel/count, "faithfulness": total_faith/count, "samples": count}
    return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate GraphRAG")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model for GraphRAG")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[Error] OPENAI_API_KEY not found in environment or ~/.env")
        sys.exit(1)
    
    print("=" * 60)
    print(f" Evaluating GraphRAG (Microsoft) using {args.model}")
    print("=" * 60)
    
    local_res = evaluate_local_dataset(args.model)
    
    with open("experiments/graphrag_results.json", "w") as f:
        json.dump({
            "model": args.model,
            "local": local_res,
        }, f, indent=2)

if __name__ == "__main__":
    main()
