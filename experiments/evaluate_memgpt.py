import os
import json
import argparse
from dotenv import load_dotenv

# Load API keys securely from ~/.env
load_dotenv(os.path.expanduser("~/.env"))

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.evaluate_generation import judge_answer
from src.preprocessing import clean_text
from src.utils import find_chapters, extract_text_upto_chapter
from src.bookmark import load_bookmark

try:
    from memgpt import create_client
except ImportError:
    print("[Error] Please install memgpt: pip install memgpt")
    sys.exit(1)

BOOK_PATH = "data/book.txt"
QA_PATH = "benchmark/qa_pairs.json"

def evaluate_local_dataset():
    print(f"\n[Eval] Loading Local Dataset (Book)")
    bookmark = load_bookmark()
    chapters = find_chapters(BOOK_PATH)
    text = extract_text_upto_chapter(BOOK_PATH, chapters, bookmark["pov"], bookmark["occurrence"])
    text = clean_text(text)

    # Initialize MemGPT
    client = create_client()
    agent = client.create_agent(name="benchmark_agent")
    
    # Send document to MemGPT archival memory
    client.send_message(agent_id=agent.id, message=f"Please store this in your archival memory:\n{text}", role="user")

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
        
        # Query MemGPT
        response = client.send_message(agent_id=agent.id, message=question, role="user")
        answer = response.messages[-1].text if response.messages else ""
        
        scores = judge_answer(question, ground_truth, answer, [text], model="qwen2:0.5b")
        total_rel += scores["relevance"]
        total_faith += scores["faithfulness"]
        count += 1
        print(f"  Q: {question}\n  -> Rel: {scores['relevance']}/5 | Faith: {scores['faithfulness']}/5")

    if count > 0:
        return {"relevance": total_rel/count, "faithfulness": total_faith/count, "samples": count}
    return None

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[Error] OPENAI_API_KEY not found in environment or ~/.env")
        sys.exit(1)
    
    print("=" * 60)
    print(f" Evaluating MemGPT (Virtual OS Memory)")
    print("=" * 60)
    
    local_res = evaluate_local_dataset()
    
    with open("experiments/memgpt_results.json", "w") as f:
        json.dump({
            "model": "memgpt-gpt4",
            "local": local_res,
        }, f, indent=2)

if __name__ == "__main__":
    main()
