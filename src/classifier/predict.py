"""
Learned Event Classifier — Inference Module
=============================================
Loads the sklearn Logistic Regression model (Linear Probe on dense embeddings)
from src/classifier/model/sklearn_model.pkl and provides classify_event_learned() 
as a drop-in replacement for the rule-based classify_event().

Auto-detection:
  - If the model exists → uses learned classifier
  - If not → falls back silently to rule-based (no crash, no warning spam)

Confidence gating:
  - If the model's confidence is below CONFIDENCE_THRESHOLD,
    the prediction falls back to the rule-based classifier.
"""

import os
import pickle
import numpy as np
from src.embeddings import embed_memories

# ── Paths ─────────────────────────────────────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "sklearn_model.pkl")
_LOAD_FAILED = object()     # Sentinel to prevent repeated load attempts

# ── Module-level state (lazy-loaded) ─────────────────────────────────────────
_model = None
CONFIDENCE_THRESHOLD = 0.45     # Below this, fall back to rules

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

LABEL_IMPORTANCE = {
    "death":        4,
    "resurrection": 4,
    "combat":       3,
    "discovery":    3,
    "dialogue":     1,
    "atmosphere":   1,
    "description":  1,
}

# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model():
    global _model
    if _model is _LOAD_FAILED:
        return False
    if _model is not None:
        return True
    if not os.path.isfile(_MODEL_PATH):
        _model = _LOAD_FAILED
        return False
    try:
        with open(_MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        print(f"[Classifier] Learned Sklearn model loaded from {_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[Classifier] Failed to load learned model: {e}")
        _model = _LOAD_FAILED
        return False

# ── Public API ────────────────────────────────────────────────────────────────

def is_available() -> bool:
    """Returns True if the fine-tuned model is available for inference."""
    return _load_model()

def classify_event_learned(sentence: str) -> tuple:
    if not _load_model():
        raise RuntimeError("Learned classifier not available")
        
    embedding = embed_memories([sentence])[0]
    X = np.array([embedding])
    probs = _model.predict_proba(X)[0]
    
    pred_class = np.argmax(probs)
    confidence = float(probs[pred_class])
    
    # Sklearn model classes are usually the labels from training
    # We mapped them to IDs in training (0,1,2,3...)
    # We just need to check the _model.classes_ array
    pred_label_id = _model.classes_[pred_class]
    event_type = ID2LABEL.get(pred_label_id, "description")
    importance = LABEL_IMPORTANCE.get(event_type, 1)

    return event_type, importance, confidence

def classify_batch(sentences: list[str]) -> list[tuple]:
    if not _load_model():
        raise RuntimeError("Learned classifier not available")
        
    embeddings = embed_memories(sentences)
    X = np.array(embeddings)
    probs = _model.predict_proba(X)
    
    results = []
    for row in probs:
        pred_class = np.argmax(row)
        confidence = float(row[pred_class])
        pred_label_id = _model.classes_[pred_class]
        event_type = ID2LABEL.get(pred_label_id, "description")
        importance = LABEL_IMPORTANCE.get(event_type, 1)
        results.append((event_type, importance, confidence))

    return results
