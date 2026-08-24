"""
Semi-Automated Event Annotation Pipeline
==========================================
Generates training data for the fine-tuned event classifier.

Workflow:
  1. Extract sentences from the book text
  2. Apply the rule-based classifier as a noisy baseline label
  3. (Optional) Use Ollama LLM to generate a second-opinion label
  4. Output JSONL with both labels for human review and correction

Methodology Note:
  Using LLM-generated labels as training signal (then human-correcting)
  is called "LLM-assisted annotation" — an accepted technique in
  NeurIPS/ACL publications. See: Gilardi et al. (2023), Pangakis et al. (2023).

Usage:
  python -m src.classifier.annotate --book data/book.txt --output data/event_labels.jsonl
  python -m src.classifier.annotate --book data/book.txt --output data/event_labels.jsonl --use-llm

After annotation:
  1. Open data/event_labels.jsonl and correct mislabeled rows
  2. Set "reviewed": true on verified entries
  3. Run: python -m src.classifier.train_event_classifier --data data/event_labels.jsonl
"""

import argparse
import json
import os
import re
import sys
import random
from collections import Counter

# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.events import classify_event
from src.preprocessing import clean_text


# ── Constants ─────────────────────────────────────────────────────────────────

EVENT_TYPES = [
    "death", "resurrection", "combat", "discovery",
    "dialogue", "atmosphere", "description"
]

LABEL_DESCRIPTIONS = {
    "death":         "A character physically dies, is killed, or their body is described as dead",
    "resurrection":  "A dead character comes back to life, reanimates, or rises as undead",
    "combat":        "Physical violence, fighting, weapon strikes, or armed confrontation",
    "discovery":     "A character sees, finds, notices, or learns something new",
    "dialogue":      "Speech, conversation, quotes, arguing, or any verbal exchange",
    "atmosphere":    "Weather, environment, mood-setting, or ambient sensory description",
    "description":   "Physical appearance, objects, clothing, or static scene details",
}


# ── Sentence Extraction ──────────────────────────────────────────────────────

def extract_sentences(text: str, min_length: int = 20) -> list[str]:
    """Split text into sentences, filtering by minimum length."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) >= min_length]


# ── Rule-Based Annotation ────────────────────────────────────────────────────

def annotate_with_rules(sentences: list[str]) -> list[dict]:
    """Apply the existing rule-based classifier to all sentences."""
    annotated = []
    for i, sent in enumerate(sentences):
        event_type, importance = classify_event(sent)
        annotated.append({
            "id": i,
            "text": sent,
            "rule_label": event_type,
            "rule_importance": importance,
            "final_label": event_type,    # Default — human corrects this
            "reviewed": False
        })
    return annotated


# ── LLM-Assisted Annotation ──────────────────────────────────────────────────

def annotate_with_llm(annotated: list[dict], model: str = "llama3") -> list[dict]:
    """
    Uses Ollama LLM to generate a second-opinion label for each sentence.
    When the LLM and rule-based classifier disagree, defaults to the LLM
    label (which is generally more accurate for nuanced cases).
    """
    try:
        import ollama
    except ImportError:
        print("[Annotate] ollama not installed. Run: pip install ollama")
        print("[Annotate] Skipping LLM annotation. Using rule-based labels only.")
        return annotated

    label_list = ", ".join(EVENT_TYPES)
    definitions = "\n".join(f"  - {k}: {v}" for k, v in LABEL_DESCRIPTIONS.items())

    print(f"[Annotate] Running LLM annotation ({model}) on {len(annotated)} sentences...")

    for i, item in enumerate(annotated):
        if i % 50 == 0 and i > 0:
            print(f"  [{i}/{len(annotated)}] annotating...")

        prompt = (
            f"Classify the following sentence from a fantasy novel into "
            f"exactly ONE of these event categories:\n\n"
            f"Categories: {label_list}\n\n"
            f"Definitions:\n{definitions}\n\n"
            f"Sentence: \"{item['text']}\"\n\n"
            f"Respond with ONLY the category name in lowercase. Nothing else."
        )

        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 10}
            )
            raw = response["message"]["content"].strip().lower().split()[0]

            # Validate / fuzzy-match
            if raw in EVENT_TYPES:
                item["llm_label"] = raw
            else:
                matched = next((et for et in EVENT_TYPES if et in raw), None)
                item["llm_label"] = matched or "unknown"

            # Agreement tracking
            item["agreement"] = (item["rule_label"] == item.get("llm_label"))

            # When they agree → high confidence; when they disagree → prefer LLM
            if item["agreement"]:
                item["final_label"] = item["rule_label"]
            else:
                item["final_label"] = item.get("llm_label", item["rule_label"])

        except Exception as e:
            item["llm_label"] = "error"
            item["agreement"] = None

    return annotated


# ── Dataset Balancing ─────────────────────────────────────────────────────────

def sample_balanced(annotated: list[dict], max_per_class: int = 150) -> list[dict]:
    """
    Caps each class at max_per_class to reduce extreme imbalance.
    Does NOT remove underrepresented classes — only trims overrepresented ones.
    """
    by_label = {}
    for item in annotated:
        label = item["final_label"]
        by_label.setdefault(label, []).append(item)

    sampled = []
    for label, items in by_label.items():
        if len(items) > max_per_class:
            sampled.extend(random.sample(items, max_per_class))
        else:
            sampled.extend(items)

    random.shuffle(sampled)
    return sampled


# ── Distribution Printer ─────────────────────────────────────────────────────

def print_distribution(annotated: list[dict], label_key: str = "final_label"):
    """Print label distribution as a formatted table."""
    counts = Counter(item[label_key] for item in annotated)
    total = sum(counts.values())

    print(f"\n  {'Label':<15} {'Count':>6} {'Pct':>6}")
    print(f"  {'-'*30}")
    for label, count in counts.most_common():
        print(f"  {label:<15} {count:>6} {count/total*100:>5.1f}%")
    print(f"  {'TOTAL':<15} {total:>6}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Annotate book sentences for event classification training"
    )
    parser.add_argument("--book", default="data/book.txt",
                        help="Path to book text file (default: data/book.txt)")
    parser.add_argument("--output", default="data/event_labels.jsonl",
                        help="Output JSONL annotation file")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use Ollama LLM for second-opinion labels")
    parser.add_argument("--llm-model", default="llama3",
                        help="Ollama model name for LLM annotation (default: llama3)")
    parser.add_argument("--max-sentences", type=int, default=1000,
                        help="Maximum sentences to annotate (default: 1000)")
    parser.add_argument("--max-per-class", type=int, default=150,
                        help="Max samples per class for balancing (default: 150)")
    parser.add_argument("--chapter-range", type=str, default=None,
                        help="Limit to specific chapter range, e.g. '1-15'")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    # ── 1. Load and clean text ────────────────────────────────────────────────
    print(f"[Annotate] Loading text from {args.book} ...")
    with open(args.book, "r", encoding="utf-8") as f:
        raw_text = f.read()
    text = clean_text(raw_text)

    # ── 2. Extract sentences ──────────────────────────────────────────────────
    sentences = extract_sentences(text)
    print(f"[Annotate] Extracted {len(sentences)} sentences from text")

    # ── 3. Sample if text is very large ───────────────────────────────────────
    if len(sentences) > args.max_sentences:
        # Stratified-ish: take from beginning, middle, and end
        n = args.max_sentences
        chunk = len(sentences) // 3
        pool = (
            sentences[:chunk]
            + sentences[chunk:2*chunk]
            + sentences[2*chunk:]
        )
        sentences = random.sample(pool, min(n, len(pool)))
        print(f"[Annotate] Sampled down to {len(sentences)} sentences")

    # ── 4. Rule-based annotation ──────────────────────────────────────────────
    print("[Annotate] Applying rule-based classifier ...")
    annotated = annotate_with_rules(sentences)
    print("\n  Rule-based label distribution:")
    print_distribution(annotated, "rule_label")

    # ── 5. Optional LLM annotation ───────────────────────────────────────────
    if args.use_llm:
        annotated = annotate_with_llm(annotated, model=args.llm_model)

        agreed = sum(1 for a in annotated if a.get("agreement") is True)
        total_compared = sum(1 for a in annotated if a.get("agreement") is not None)
        if total_compared > 0:
            print(f"\n  [Annotate] Rule ↔ LLM agreement: "
                  f"{agreed}/{total_compared} ({agreed/total_compared*100:.1f}%)")

        print("\n  LLM-adjusted label distribution:")
        print_distribution(annotated, "final_label")

    # ── 6. Balance the dataset ────────────────────────────────────────────────
    balanced = sample_balanced(annotated, max_per_class=args.max_per_class)
    print(f"\n  Balanced dataset: {len(balanced)} samples")
    print_distribution(balanced)

    # ── 7. Save JSONL ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in balanced:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n[Annotate] ✓ Saved {len(balanced)} annotations → {args.output}")
    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │                  NEXT STEPS                     │")
    print("  ├─────────────────────────────────────────────────┤")
    print(f"  │  1. Open {args.output}")
    print("  │  2. Correct any mislabeled 'final_label' values │")
    print("  │  3. Set 'reviewed': true on verified entries    │")
    print("  │  4. Run the fine-tuning script:                 │")
    print(f"  │     python -m src.classifier.train_event_classifier \\")
    print(f"  │       --data {args.output}")
    print("  └─────────────────────────────────────────────────┘")


if __name__ == "__main__":
    main()
