"""
Event Classifier — Fine-Tuning Script
=======================================
Fine-tunes a transformer model on the annotated event classification dataset.

Supported models:
  - distilbert  → distilbert-base-uncased  (fast, good for M3 Mac MPS)
  - deberta     → microsoft/deberta-v3-small (more accurate, slightly slower)

Features:
  - Stratified train / validation / test split (70 / 15 / 15)
  - Class-weighted cross-entropy loss for handling label imbalance
  - Per-class F1, precision, recall + macro averages
  - Confusion matrix visualization (saved to experiments/)
  - Early stopping on validation macro-F1
  - Model + tokenizer saved to src/classifier/model/

Usage:
  python -m src.classifier.train_event_classifier --data data/event_labels.jsonl
  python -m src.classifier.train_event_classifier --data data/event_labels.jsonl --model deberta --epochs 15

Requirements:
  pip install torch transformers datasets scikit-learn accelerate matplotlib
"""

import argparse
import json
import os
import sys
import random
import numpy as np
from collections import Counter

# ── Label configuration ──────────────────────────────────────────────────────

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

MODEL_MAP = {
    "distilbert": "sentence-transformers/all-MiniLM-L6-v2",
    "deberta":    "microsoft/deberta-v3-small",
}

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "model")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_annotations(path: str) -> list[dict]:
    """Load JSONL annotation file. Uses 'final_label' as the ground truth."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            label = item.get("final_label", item.get("rule_label", "description"))
            if label not in LABEL2ID:
                continue    # Skip unknown labels
            data.append({
                "text": item["text"],
                "label": label,
                "label_id": LABEL2ID[label],
            })
    return data


def stratified_split(data: list[dict], train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Stratified split into train / val / test sets.
    Ensures each label is proportionally represented in each split.
    """
    rng = random.Random(seed)
    by_label = {}
    for item in data:
        by_label.setdefault(item["label"], []).append(item)

    train, val, test = [], [], []

    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


# ── Compute class weights for imbalanced data ────────────────────────────────

def compute_class_weights(data: list[dict]) -> list[float]:
    """
    Inverse frequency weighting: w_c = N / (K * n_c)
    where N = total samples, K = number of classes, n_c = samples in class c.
    """
    counts = Counter(item["label_id"] for item in data)
    total = sum(counts.values())
    n_classes = len(LABEL2ID)

    weights = []
    for i in range(n_classes):
        n_c = counts.get(i, 1)    # Avoid division by zero
        weights.append(total / (n_classes * n_c))

    return weights


# ── Main training function ───────────────────────────────────────────────────

def train(args):
    # Late imports so the script shows --help without torch installed
    import torch
    from torch.utils.data import Dataset as TorchDataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
    )

    # ── Device selection ──────────────────────────────────────────────────────
    # Force CPU to prevent MPS deadlock during model download/tokenizer phase
    device = "cpu"
    print(f"[Train] Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"[Train] Loading annotations from {args.data} ...")
    data = load_annotations(args.data)
    print(f"[Train] Loaded {len(data)} samples")

    if len(data) < 30:
        print("[Train] ERROR: Need at least 30 annotated samples to fine-tune.")
        print("[Train] Run the annotation pipeline first:")
        print("  python -m src.classifier.annotate --book data/book.txt --output data/event_labels.jsonl")
        sys.exit(1)

    train_data, val_data, test_data = stratified_split(data, seed=args.seed)
    print(f"[Train] Split: {len(train_data)} train / {len(val_data)} val / {len(test_data)} test")

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weights = compute_class_weights(train_data)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"[Train] Class weights: { {ID2LABEL[i]: f'{w:.2f}' for i, w in enumerate(class_weights)} }")

    # ── Tokenizer + Model ─────────────────────────────────────────────────────
    model_name = MODEL_MAP.get(args.model, args.model)
    print(f"[Train] Loading {model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    # ── Dataset class ─────────────────────────────────────────────────────────
    class EventDataset(TorchDataset):
        def __init__(self, items, tokenizer, max_length=128):
            self.items = items
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            item = self.items[idx]
            encoding = self.tokenizer(
                item["text"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids":      encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels":         torch.tensor(item["label_id"], dtype=torch.long),
            }

    train_dataset = EventDataset(train_data, tokenizer)
    val_dataset = EventDataset(val_data, tokenizer)
    test_dataset = EventDataset(test_data, tokenizer)

    # ── Custom Trainer with weighted loss ─────────────────────────────────────
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # ── Metrics function ──────────────────────────────────────────────────────
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        return {"macro_f1": macro_f1, "weighted_f1": weighted_f1}

    # ── Training arguments ────────────────────────────────────────────────────
    output_dir = args.output or DEFAULT_OUTPUT_DIR
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        save_total_limit=2,
        report_to="none",        # No wandb/tensorboard unless configured
        fp16=False,              # MPS doesn't support fp16 well
        dataloader_num_workers=0,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[Train] Starting fine-tuning for {args.epochs} epochs ...")
    trainer.train()

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("\n[Train] Evaluating on held-out test set ...")
    test_preds = trainer.predict(test_dataset)
    pred_labels = np.argmax(test_preds.predictions, axis=-1)
    true_labels = np.array([item["label_id"] for item in test_data])

    # Classification report
    report = classification_report(
        true_labels, pred_labels,
        target_names=[ID2LABEL[i] for i in range(len(ID2LABEL))],
        zero_division=0
    )
    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT (Test Set)")
    print("=" * 60)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    print("  CONFUSION MATRIX:")
    labels_str = [ID2LABEL[i][:8] for i in range(len(ID2LABEL))]
    header = "         " + " ".join(f"{l:>8}" for l in labels_str)
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join(f"{v:>8}" for v in row)
        print(f"  {labels_str[i]:>8} {row_str}")

    # ── Save confusion matrix plot ────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title("Event Classifier — Confusion Matrix (Test Set)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        tick_labels = [ID2LABEL[i] for i in range(len(ID2LABEL))]
        ax.set_xticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(tick_labels)))
        ax.set_yticklabels(tick_labels)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        plt.tight_layout()
        cm_path = os.path.join("experiments", "confusion_matrix.png")
        os.makedirs("experiments", exist_ok=True)
        plt.savefig(cm_path, dpi=150)
        print(f"\n[Train] Confusion matrix saved → {cm_path}")
    except ImportError:
        print("[Train] matplotlib not installed — skipping confusion matrix plot.")

    # ── Save model + tokenizer ────────────────────────────────────────────────
    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Save label mapping alongside the model
    meta = {
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "model_name": model_name,
        "test_macro_f1": float(test_preds.metrics.get("test_macro_f1", 0)),
        "test_weighted_f1": float(test_preds.metrics.get("test_weighted_f1", 0)),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
    }
    with open(os.path.join(final_path, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Train] ✓ Model + tokenizer saved → {final_path}")
    print(f"[Train] ✓ Test macro-F1: {meta['test_macro_f1']:.4f}")
    print(f"[Train] ✓ Test weighted-F1: {meta['test_weighted_f1']:.4f}")

    print("\n  ┌────────────────────────────────────────────────────┐")
    print("  │  Fine-tuning complete!                             │")
    print("  │                                                    │")
    print("  │  The learned classifier is now active.              │")
    print("  │  events.py will auto-detect and use it when you    │")
    print("  │  run main.py or the benchmark.                     │")
    print("  └────────────────────────────────────────────────────┘")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a transformer on event classification"
    )
    parser.add_argument("--data", default="data/event_labels.jsonl",
                        help="Path to annotated JSONL file")
    parser.add_argument("--model", default="distilbert",
                        choices=list(MODEL_MAP.keys()),
                        help="Base model to fine-tune (default: distilbert)")
    parser.add_argument("--output", default=None,
                        help=f"Output directory for model (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate (default: 3e-5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()
