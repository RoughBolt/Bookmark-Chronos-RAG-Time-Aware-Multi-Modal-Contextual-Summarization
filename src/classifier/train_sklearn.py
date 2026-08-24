"""
Event Classifier — Fast Embedding + Logistic Regression
=========================================================
Trains a Logistic Regression model on top of nomic-embed-text-v1 embeddings.
This avoids heavy PyTorch transformer fine-tuning and provides an extremely
fast, robust classifier for Apple Silicon architectures.
"""

import json
import os
import pickle
import numpy as np

# Prevent macOS multiprocessing deadlocks with tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.embeddings import embed_memories

LABEL2ID = {
    "death":        0,
    "resurrection": 1,
    "combat":       2,
    "discovery":    3,
    "dialogue":     4,
    "atmosphere":   5,
    "description":  6,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "model")

def load_data(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            label = item.get("final_label", item.get("rule_label", "description"))
            if label in LABEL2ID:
                texts.append(item["text"])
                labels.append(LABEL2ID[label])
    return texts, labels

def main():
    print("[Train] Loading annotations...")
    texts, labels = load_data("data/event_labels.jsonl")
    
    print(f"[Train] Generating embeddings for {len(texts)} sentences...")
    # Generate embeddings using nomic-embed-text
    embeddings = embed_memories(texts)
    X = np.array(embeddings)
    y = np.array(labels)
    
    print("[Train] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("[Train] Training Logistic Regression model (Linear Probe)...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    
    print("[Train] Evaluating on Test Set...")
    preds = model.predict(X_test)
    
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    target_names = [ID2LABEL[i] for i in range(len(ID2LABEL)) if i in y]
    print(classification_report(y_test, preds, labels=[i for i in range(len(ID2LABEL)) if i in y], target_names=target_names))
    
    # Save model
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(DEFAULT_OUTPUT_DIR, "sklearn_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
        
    print(f"\n[Train] ✓ Model saved to {out_path}")
    print("  The learned classifier is now active and uses dense embeddings!")

if __name__ == "__main__":
    main()
