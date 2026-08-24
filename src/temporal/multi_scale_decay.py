"""
Multi-Scale Temporal Decay Model
==================================
Cognitive science-inspired memory decay based on the Atkinson & Shiffrin (1968)
multi-store model of human memory, adapted for narrative event retrieval.

Theory:
  Human memory operates at multiple timescales through distinct stores:

    ┌─────────────────┐     promotion      ┌─────────────────┐     consolidation    ┌─────────────────┐
    │  EPISODIC BUFFER │ ─────────────────► │  WORKING MEMORY  │ ──────────────────► │  LONG-TERM STORE │
    │  (fast decay)    │  importance ≥ 3    │  (medium decay)  │  importance ≥ 4     │  (slow decay)    │
    │  λ ~ 1.5 – 2.0  │                    │  λ ~ 0.3 – 0.6  │  or high salience   │  λ ~ 0.05 – 0.1  │
    └─────────────────┘                    └─────────────────┘                     └─────────────────┘
           ▲                                                                               ▲
           │                                                                               │
    dialogue, atmosphere,                                                           death, resurrection,
    description events                                                              character-defining events

    ┌─────────────────────────┐
    │  SEMANTIC MEMORY         │  ← Knowledge graph (character statuses, interactions)
    │  (zero decay / permanent)│  ← Already handled by knowledge/graph_builder.py
    └─────────────────────────┘

Key innovation over single-scale decay:
  - Single-scale applies ONE λ per event type → "combat" always decays at 0.5
  - Multi-scale allows the SAME event to decay differently based on its
    narrative position, emotional salience, and retrieval history
  - Events can be "promoted" to slower-decaying stores based on downstream
    importance signals, mimicking memory consolidation

Mathematical formulation:

  Single-scale (current):
    S = S_base · e^(−λ_type · Δt)

  Multi-scale (this module):
    S = S_base · e^(−λ_store · Δt) · P(store | event)

    Where:
      λ_store = decay rate of the assigned memory store
      P(store | event) = probability of store assignment based on:
        - event type
        - base importance score
        - positional salience (proximity to narrative peaks)
        - emotional valence indicators

References:
  - Atkinson, R.C. & Shiffrin, R.M. (1968). "Human memory: A proposed system
    and its control processes." Psychology of Learning and Motivation, 2, 89-195.
  - Ebbinghaus, H. (1885). "Über das Gedächtnis."
  - Rubin, D.C. & Wenzel, A.E. (1996). "One hundred years of forgetting."
    Psychological Review, 103(4), 734.
"""

import math
from enum import Enum
from collections import Counter


# ── Memory store definitions ─────────────────────────────────────────────────

class MemoryStore(Enum):
    """The four memory stores, ordered by decay speed (fastest → slowest)."""
    EPISODIC  = "episodic"      # Fast decay — fleeting details
    WORKING   = "working"       # Medium decay — active narrative events
    LONG_TERM = "long_term"     # Slow decay — significant plot points
    SEMANTIC  = "semantic"      # No decay — factual state (knowledge graph)


# Default λ per store — these can be overridden by the optimizer (Phase 2A)
STORE_LAMBDA = {
    MemoryStore.EPISODIC:  1.8,     # Rapid forgetting
    MemoryStore.WORKING:   0.45,    # Gradual fade
    MemoryStore.LONG_TERM: 0.08,    # Near-permanent for narrative timescales
    MemoryStore.SEMANTIC:  0.0,     # Never decays (handled by knowledge graph)
}


# ── Store assignment rules ───────────────────────────────────────────────────

# Event types that are inherently long-term memory anchors
LONG_TERM_TYPES = {"death", "resurrection"}

# Event types that are inherently episodic (fast-fading)
EPISODIC_TYPES = {"dialogue", "atmosphere", "description"}

# Everything else (combat, discovery) goes to working memory by default


# ── Narrative salience indicators ─────────────────────────────────────────────

# Keywords that signal emotionally salient events (promote to higher store)
EMOTIONAL_MARKERS = {
    "killed", "died", "murdered", "betrayed", "wept", "screamed",
    "blood", "fire", "sword", "crowned", "married", "oath",
    "father", "mother", "brother", "sister", "son", "daughter",
    "promise", "secret", "war", "throne", "dragon",
}


def _emotional_salience(text: str) -> float:
    """
    Computes a [0.0, 1.0] emotional salience score based on
    the presence of emotionally charged narrative markers.
    """
    words = set(text.lower().split())
    matches = words & EMOTIONAL_MARKERS
    # Sigmoid-like mapping: 0 matches → 0.0, 3+ matches → ~1.0
    raw = len(matches) / 3.0
    return min(1.0, raw)


def _positional_salience(position: int, max_position: int) -> float:
    """
    Events near narrative peaks (chapter endings, climactic moments)
    are more likely to be remembered. This uses a simple recency bias:
    events closer to the bookmark (recent reading) score higher.

    Returns [0.0, 1.0] where 1.0 = at the bookmark (most recent).
    """
    if max_position == 0:
        return 1.0
    return 1.0 - (abs(position - max_position) / max_position)


# ── Core: Store assignment ────────────────────────────────────────────────────

def assign_store(
    event: dict,
    max_position: int = 1,
    promotion_threshold: float = 0.6
) -> MemoryStore:
    """
    Assigns an event to a memory store based on multi-factor analysis.

    Factors considered:
      1. Event type (death → long-term, dialogue → episodic)
      2. Base importance score (≥4 → long-term, ≥3 → working)
      3. Emotional salience of the text
      4. Positional salience (recency bias)

    The combined salience score can PROMOTE an event to a higher store:
      - Episodic event with high emotional salience → Working
      - Working event with importance ≥ 4 + high salience → Long-Term

    Args:
        event: Event dict with 'type', 'importance', 'text', 'position'.
        max_position: Maximum event position (for normalization).
        promotion_threshold: Combined salience score above which promotion occurs.

    Returns:
        MemoryStore assignment for this event.
    """
    event_type = event.get("type", "description").lower()
    importance = event.get("importance", 1)
    text = event.get("text", "")
    position = event.get("position", 0)

    # ── Step 1: Base assignment by event type ─────────────────────────────────
    if event_type in LONG_TERM_TYPES:
        base_store = MemoryStore.LONG_TERM
    elif event_type in EPISODIC_TYPES:
        base_store = MemoryStore.EPISODIC
    else:
        base_store = MemoryStore.WORKING

    # ── Step 2: Override by importance ────────────────────────────────────────
    if importance >= 4:
        base_store = MemoryStore.LONG_TERM
    elif importance >= 3 and base_store == MemoryStore.EPISODIC:
        base_store = MemoryStore.WORKING

    # ── Step 3: Promotion via salience ────────────────────────────────────────
    e_salience = _emotional_salience(text)
    p_salience = _positional_salience(position, max_position)

    # Weighted combination: emotional salience matters more than positional
    combined_salience = 0.7 * e_salience + 0.3 * p_salience

    if combined_salience >= promotion_threshold:
        if base_store == MemoryStore.EPISODIC:
            base_store = MemoryStore.WORKING
        elif base_store == MemoryStore.WORKING:
            base_store = MemoryStore.LONG_TERM
        # Long-term cannot be promoted further (semantic is KG-only)

    return base_store


# ── Core: Multi-scale decay computation ──────────────────────────────────────

def apply_multi_scale_decay(
    events: list,
    store_lambdas: dict = None,
    promotion_threshold: float = 0.6
) -> list:
    """
    Applies the multi-scale temporal decay model to a list of events.

    Each event is assigned to a memory store (episodic/working/long-term),
    and decayed using that store's λ value. This produces a more cognitively
    faithful importance distribution than single-λ-per-type decay.

    Formula:
      S_final = S_base · e^(−λ_store · Δt)

    Where λ_store depends on the STORE the event is assigned to (not just its type).

    Args:
        events: List of event dicts (must have 'importance', 'position', 'type', 'text').
        store_lambdas: Optional override dict {MemoryStore: float} for λ values.
                       If None, uses STORE_LAMBDA defaults.
        promotion_threshold: Salience threshold for store promotion (default 0.6).

    Returns:
        Same list with added fields:
          - 'decay_score': decayed importance score
          - 'memory_store': assigned MemoryStore name
          - 'store_lambda': λ value used for this event
    """
    if not events:
        return events

    lambdas = store_lambdas or STORE_LAMBDA

    # Normalization reference
    bookmark_position = max(e["position"] for e in events)
    max_position = bookmark_position if bookmark_position > 0 else 1

    # Store distribution tracking (for reporting)
    store_counts = Counter()

    for e in events:
        # Assign to memory store
        store = assign_store(e, max_position, promotion_threshold)
        base_lam = lambdas.get(store, 1.0)

        # Compute normalized temporal distance
        distance = abs(e["position"] - bookmark_position)
        delta_t = distance / max_position      # [0.0, 1.0]

        # Calculate dynamic lambda scaling based on salience
        e_salience = _emotional_salience(e.get("text", ""))
        p_salience = _positional_salience(e.get("position", 0), max_position)
        combined_salience = 0.7 * e_salience + 0.3 * p_salience
        
        # Highly salient events within a store decay up to 50% slower
        lam = base_lam * (1.0 - (0.5 * combined_salience))

        # Apply exponential decay with dynamically scaled λ
        e["decay_score"] = e["importance"] * math.exp(-lam * delta_t)
        e["memory_store"] = store.value
        e["store_lambda"] = lam

        store_counts[store.value] += 1

    # Report store distribution
    total = sum(store_counts.values())
    dist_str = ", ".join(
        f"{store}: {count} ({count/total*100:.0f}%)"
        for store, count in sorted(store_counts.items())
    )
    print(f"[Memory] Multi-scale store assignment: {dist_str}")

    return events


# ── Utility: Compare single-scale vs multi-scale ─────────────────────────────

def compare_decay_models(events: list, single_scale_lambdas: dict = None) -> dict:
    """
    Runs both single-scale and multi-scale decay on the same events and
    returns comparison statistics. Useful for ablation studies.

    Args:
        events: List of event dicts.
        single_scale_lambdas: The original per-type λ dict from events.py.

    Returns:
        Dict with comparison metrics:
          - 'single_scale_mean': mean decay_score under single-scale
          - 'multi_scale_mean': mean decay_score under multi-scale
          - 'survival_rate_single': fraction of events with decay_score > 0.5
          - 'survival_rate_multi': fraction of events with decay_score > 0.5
          - 'store_distribution': Counter of store assignments
          - 'promotion_count': number of events promoted to higher stores
    """
    import copy

    if single_scale_lambdas is None:
        single_scale_lambdas = {
            "death": 0.1, "resurrection": 0.1, "combat": 0.5,
            "discovery": 0.4, "dialogue": 1.2, "atmosphere": 2.0,
            "description": 2.0,
        }

    # Deep copy to avoid mutation
    events_single = copy.deepcopy(events)
    events_multi = copy.deepcopy(events)

    # Apply single-scale
    bookmark_pos = max(e["position"] for e in events_single)
    max_pos = bookmark_pos if bookmark_pos > 0 else 1
    for e in events_single:
        dist = abs(e["position"] - bookmark_pos)
        dt = dist / max_pos
        lam = single_scale_lambdas.get(e["type"], 1.0)
        e["decay_score"] = e["importance"] * math.exp(-lam * dt)

    # Apply multi-scale
    events_multi = apply_multi_scale_decay(events_multi)

    # Compute stats
    scores_s = [e["decay_score"] for e in events_single]
    scores_m = [e["decay_score"] for e in events_multi]

    survive_s = sum(1 for s in scores_s if s > 0.5) / len(scores_s) if scores_s else 0
    survive_m = sum(1 for s in scores_m if s > 0.5) / len(scores_m) if scores_m else 0

    store_dist = Counter(e.get("memory_store", "unknown") for e in events_multi)

    # Count promotions (events whose store is higher than their type would suggest)
    promotions = 0
    for e in events_multi:
        etype = e.get("type", "description").lower()
        store = e.get("memory_store", "episodic")
        if etype in EPISODIC_TYPES and store != "episodic":
            promotions += 1
        elif etype not in LONG_TERM_TYPES and etype not in EPISODIC_TYPES and store == "long_term":
            promotions += 1

    return {
        "single_scale_mean": sum(scores_s) / len(scores_s) if scores_s else 0,
        "multi_scale_mean": sum(scores_m) / len(scores_m) if scores_m else 0,
        "survival_rate_single": survive_s,
        "survival_rate_multi": survive_m,
        "store_distribution": dict(store_dist),
        "promotion_count": promotions,
        "total_events": len(events),
    }
