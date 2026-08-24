# ===========================VERSION 1===========================

# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.tag import pos_tag

# def extract_events(text):
#     sentences = sent_tokenize(text)
#     events = []

#     for sent in sentences:
#         words = word_tokenize(sent)
#         events = []

#         for sent in sentences:
#             words = word_tokenize(sent)
#             tagged = pos_tag(words)

#             has_verb = any(tag.startswith('VB') for _, tag in tagged)

#             if has_verb:
#                 events.append(sent)
        
#     return events

# ===========================VERSION 2===========================

import re
import math
import os

# ── Classifier mode flag ──────────────────────────────────────────────────────
# When True, extract_events() will attempt to use the fine-tuned transformer
# classifier. If the model isn't available, it falls back to rules silently.
USE_LEARNED_CLASSIFIER = True

# ── Decay model flag ─────────────────────────────────────────────────────────
# When True, apply_temporal_decay() uses the Atkinson-Shiffrin multi-store
# model (episodic/working/long-term stores). When False, uses the original
# single-scale per-type λ decay.
USE_MULTI_SCALE_DECAY = True

# ── Temporal Decay Constants (λ) — Ebbinghaus Forgetting Curve calibrated ───
#
# Formula: S = S_semantic · e^(−λ · Δt)
#   S_semantic = base importance score from classify_event() [1–4]
#   Δt         = normalized positional distance from bookmark [0.0 → 1.0]
#   λ          = decay constant tuned per event type
#
# Justification:
#   - Deaths/resurrections are long-term memory anchors → low λ (slow decay)
#   - Combat/discovery are vivid but fade at medium rate → mid λ
#   - Dialogue/atmosphere have near-zero recall value after hours → high λ
#   This mirrors the differential retention rates studied by Ebbinghaus (1885).
#
DECAY_LAMBDA = {
    "death":        0.1,
    "resurrection": 0.1,
    "combat":       0.5,
    "discovery":    0.4,
    "dialogue":     1.2,
    "atmosphere":   2.0,
    "description":  2.0,
}
DECAY_LAMBDA_DEFAULT = 1.0

def classify_event(sentence):
    s = sentence.lower()

    # TRUE death (action-based)
    # Dialogue guard
    if '"' in sentence:
        return "dialogue", 1

    if (
        any(word in s for word in ["fell", "lay", "collapsed"]) and
        any(word in s for word in ["body", "blood", "snow", "ground"])
    ):
        return "death", 4

    # The Jon Snow Edge Case (Resurrection detection)
    if any(word in s for word in ["rose", "stirred", "stood up", "awoke"]) and \
       any(word in s for word in ["dead", "body", "corpse", "pale", "eyes"]):
        return "resurrection", 4

    # Talking ABOUT death ≠ death
    if "dead" in s and not any(word in s for word in ["fell", "killed", "slain"]):
        return "dialogue", 1
    
    # Atmosphere (lowest importance)
    if any(word in s for word in ["wind", "trees", "rustle", "cold", "snow"]) and \
        not any(word in s for word in ["strike", "slash", "attack", "hit"]):
        return "atmosphere", 1
    
    if any(word in s for word in ["sword", "blade", "iron"]) and \
        not any(word in s for word in ["raised", "swung", "slashed"]):
        return "description", 1

    if any(word in s for word in ["slashed", "struck", "hit", "checked", "fell back"]):
        return "combat", 3

    if any(word in s for word in ["sword", "blow", "blade", "strike"]):
        return "combat", 3

    if any(word in s for word in ["saw", "found", "noticed", "appeared"]):
        return "discovery", 3

    if '"' in sentence:
        return "dialogue", 2

    return "description", 1


# ── Legacy deduplication (string prefix matching) ─────────────────────────────

def deduplicate_events_legacy(events):
    """Original deduplication using first-50-chars fingerprint."""
    seen = set()
    unique = []

    for e in events:
        key = e["text"][:50]  # rough string fingerprint
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique


# ── Semantic deduplication (embedding cosine similarity clustering) ────────────

def _cosine_sim(v1: list, v2: list) -> float:
    """Cosine similarity between two vectors (no numpy dependency)."""
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = sum(a * a for a in v1) ** 0.5
    n2 = sum(b * b for b in v2) ** 0.5
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def deduplicate_events_semantic(
    events: list,
    similarity_threshold: float = 0.92
) -> list:
    """
    Semantic deduplication via embedding cosine similarity.

    Algorithm (greedy leader clustering):
      1. Sort events by importance (descending) so the strongest event in
         each near-duplicate cluster is always the one kept.
      2. For each event, embed it and compare to all existing cluster leaders.
      3. If cosine similarity with any leader exceeds the threshold, skip it
         (it's a near-duplicate of a more important event).
      4. Otherwise, keep it and register it as a new cluster leader.

    Metrics reported:
      - Compression ratio: (original - deduplicated) / original
      - Events retained: absolute count

    Args:
        events: List of event dicts with 'text' and 'importance' fields.
        similarity_threshold: Cosine similarity above which two events are
                              considered duplicates (default 0.92).

    Returns:
        Deduplicated list of events (subset of the input, original order).
    """
    if len(events) <= 1:
        return events

    try:
        from src.embeddings import embed_memories
    except ImportError:
        print("[Dedup] Embeddings unavailable — falling back to legacy dedup")
        return deduplicate_events_legacy(events)

    # Batch-embed all event texts
    texts = [e["text"] for e in events]
    try:
        embeddings = embed_memories(texts)
    except Exception as e:
        print(f"[Dedup] Embedding failed ({e}) — falling back to legacy dedup")
        return deduplicate_events_legacy(events)

    # Sort indices by importance (descending) — most important kept first
    sorted_indices = sorted(
        range(len(events)),
        key=lambda i: events[i].get("importance", 1),
        reverse=True
    )

    # Greedy leader clustering
    leader_embeddings: list = []    # Embeddings of kept events
    kept_indices: set = set()       # Indices of events we're keeping

    for idx in sorted_indices:
        vec = embeddings[idx]
        is_duplicate = False

        for leader_vec in leader_embeddings:
            sim = _cosine_sim(vec, leader_vec)
            if sim >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            leader_embeddings.append(vec)
            kept_indices.add(idx)

    # Preserve original chronological order
    deduplicated = [events[i] for i in sorted(kept_indices)]

    # Report metrics
    original_count = len(events)
    dedup_count = len(deduplicated)
    compression = (original_count - dedup_count) / original_count * 100 if original_count > 0 else 0

    print(f"[Dedup] Semantic deduplication: {original_count} → {dedup_count} events "
          f"({compression:.1f}% compression, threshold={similarity_threshold})")

    return deduplicated


# Backward-compatible alias
def deduplicate_events(events):
    """Routes to semantic dedup if embeddings are available, else legacy."""
    try:
        return deduplicate_events_semantic(events)
    except Exception:
        return deduplicate_events_legacy(events)


# ── Event extraction (with learned classifier support) ────────────────────────

def extract_events(text):
    """
    Extracts and classifies events from text.

    Classifier routing:
      1. If USE_LEARNED_CLASSIFIER is True and the fine-tuned model exists,
         uses the transformer classifier with confidence gating.
      2. If the model's confidence is below threshold (0.45), falls back
         to the rule-based classifier for that specific sentence.
      3. If the model isn't available at all, uses rules for everything.

    Deduplication:
      Uses semantic (embedding-based) dedup when available, else string-prefix.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    events = []

    # ── Attempt to load learned classifier ────────────────────────────────────
    learned_available = False
    learned_classify = None
    learned_batch_classify = None

    if USE_LEARNED_CLASSIFIER:
        try:
            from src.classifier.predict import (
                is_available,
                classify_event_learned,
                classify_batch,
                CONFIDENCE_THRESHOLD,
            )
            if is_available():
                learned_available = True
                learned_classify = classify_event_learned
                learned_batch_classify = classify_batch
                print("[Events] Using fine-tuned transformer classifier")
        except ImportError:
            pass    # Model package not available — use rules

    if not learned_available:
        print("[Events] Using rule-based classifier")

    # ── Classify each sentence ────────────────────────────────────────────────
    valid_sentences = []
    valid_indices = []

    for idx, sent in enumerate(sentences):
        sent = sent.strip()
        if len(sent) < 20:
            continue
        valid_sentences.append(sent)
        valid_indices.append(idx)

    # Try batch classification with learned model
    if learned_available and learned_batch_classify and len(valid_sentences) > 0:
        try:
            batch_results = learned_batch_classify(valid_sentences)
            rule_fallback_count = 0

            for sent, idx, (etype, imp, conf) in zip(
                valid_sentences, valid_indices, batch_results
            ):
                # Confidence gating: fall back to rules if uncertain
                if conf < CONFIDENCE_THRESHOLD:
                    etype, imp = classify_event(sent)
                    rule_fallback_count += 1

                events.append({
                    "text": sent,
                    "type": etype,
                    "importance": imp,
                    "position": idx,
                    "classifier": "learned" if conf >= CONFIDENCE_THRESHOLD else "rule_fallback",
                })

            if rule_fallback_count > 0:
                print(f"[Events] Confidence fallback: {rule_fallback_count}/{len(valid_sentences)} "
                      f"sentences routed to rule-based classifier")

        except Exception as e:
            print(f"[Events] Batch classification failed ({e}), using rule-based")
            events = []    # Reset and fall through to rule-based below
            learned_available = False

    # Fall-through: rule-based classification
    if not events:
        for sent, idx in zip(valid_sentences, valid_indices):
            event_type, importance = classify_event(sent)
            events.append({
                "text": sent,
                "type": event_type,
                "importance": importance,
                "position": idx,
                "classifier": "rule",
            })

    events = deduplicate_events(events)
    return events

def get_event_threshold(days_gap):
    if days_gap > 90:
        return 4      # only critical events
    elif days_gap > 30:
        return 3
    elif days_gap > 7:
        return 2
    else:
        return 1      # everything


def get_event_limit(days_gap):
    if days_gap > 90:
        return 3
    elif days_gap > 30:
        return 5
    elif days_gap > 7:
        return 8
    else:
        return 15


def apply_temporal_decay(events: list, force_single_scale: bool = False) -> list:
    """
    Applies temporal decay to event importance scores.

    Routing:
      - If USE_MULTI_SCALE_DECAY is True (and not force_single_scale),
        uses the Atkinson-Shiffrin multi-store model from
        src/temporal/multi_scale_decay.py.
      - Otherwise, uses the original single-scale per-type λ decay.

    Single-scale formula:
      S = S_base · e^(−λ_type · Δt)

    Multi-scale formula:
      S = S_base · e^(−λ_store · Δt)   where store ∈ {episodic, working, long_term}

    Args:
        events: List of event dicts with 'importance', 'position', 'type', 'text'.
        force_single_scale: If True, bypasses USE_MULTI_SCALE_DECAY flag.

    Returns:
        Same list with 'decay_score' populated (and 'memory_store' if multi-scale).
    """
    if not events:
        return events

    # ── Multi-scale path ──────────────────────────────────────────────────────
    if USE_MULTI_SCALE_DECAY and not force_single_scale:
        try:
            from src.temporal.multi_scale_decay import apply_multi_scale_decay
            return apply_multi_scale_decay(events)
        except ImportError:
            pass    # Fall through to single-scale
        except Exception as e:
            print(f"[Decay] Multi-scale failed ({e}), using single-scale")

    # ── Single-scale path (original) ──────────────────────────────────────────
    bookmark_position = max(e["position"] for e in events)
    max_position = bookmark_position if bookmark_position > 0 else 1

    for e in events:
        distance = abs(e["position"] - bookmark_position)
        delta_t = distance / max_position          # Normalize to [0.0, 1.0]
        lam = DECAY_LAMBDA.get(e["type"], DECAY_LAMBDA_DEFAULT)

        # Exponential decay: recent events retain full score, old ones decay fast
        e["decay_score"] = e["importance"] * math.exp(-lam * delta_t)
        e["memory_store"] = "single_scale"    # For ablation tracking

    return events