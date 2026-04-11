# src/scene_abstraction.py

# ------------------------- VERSION 1 -------------------------

def old_summarize_scene(scene):
    # pick most important event as representative
    key_event = max(scene, key=lambda x: x["importance"])
    return {
        "type": key_event["type"],
        "text": key_event["text"]
    }

# ------------------------- VERSION 2 -------------------------

from collections import Counter

# -------------------------
# Template banks
# -------------------------

DIALOGUE_TEMPLATES = [
    "They argued about whether to proceed as conditions worsened.",
    "A tense debate unfolded among the group.",
    "Warnings were exchanged, but doubts remained unresolved."
]

DISCOVERY_TEMPLATES = [
    "Signs of something unnatural appeared nearby.",
    "Something felt wrong in their surroundings.",
    "An unsettling discovery revealed itself."
]

COMBAT_TEMPLATES = [
    "A sudden confrontation erupted.",
    "The encounter quickly turned violent.",
]

COMBAT_DEATH_TEMPLATES = [
    "The confrontation ended in death.",
    "The fight proved fatal."
]

# -------------------------
# Helper functions
# -------------------------

def dominant_event_type(scene_events):
    counts = Counter(e["type"] for e in scene_events)
    return counts.most_common(1)[0][0]

def contains_event(scene_events, event_type):
    return any(e["type"] == event_type for e in scene_events)

# -------------------------
# CORE FUNCTION
# -------------------------

def summarize_scene(scene_events, days_gap):
    """
    Converts a list of low-level events into ONE abstract human-like memory.
    """

    # Safety check
    if not scene_events:
        return None

    types = [e["type"] for e in scene_events]
    dominant = dominant_event_type(scene_events)

    has_discovery = contains_event(scene_events, "discovery")
    has_combat = contains_event(scene_events, "combat")
    has_death = contains_event(scene_events, "death")

    # -------------------------
    # TIME-BASED MEMORY DECAY
    # -------------------------

    # After very long gaps, only deaths survive
    if days_gap > 90 and not has_death:
        return None

    # -------------------------
    # DEATH OVERRIDES EVERYTHING
    # -------------------------

    if has_death:
        # Extract the most important death sentence
        death_event = max(
            (e for e in scene_events if e["type"] == "death"),
            key=lambda x: x["importance"]
        )

        return {
            "type": "death",
            "text": death_event["text"]
        }

    # -------------------------
    # DISCOVERY → COMBAT
    # -------------------------

    if has_discovery and has_combat:
        return {
            "type": "combat",
            "text": "A disturbing discovery led to a sudden violent confrontation."
        }

    # -------------------------
    # PURE COMBAT
    # -------------------------

    if dominant == "combat":
        return {
            "type": "combat",
            "text": COMBAT_TEMPLATES[0]
        }

    # -------------------------
    # DISCOVERY-FOCUSED SCENE
    # -------------------------

    if dominant == "discovery":
        return {
            "type": "discovery",
            "text": DISCOVERY_TEMPLATES[0]
        }

    # -------------------------
    # DIALOGUE-HEAVY SCENE
    # -------------------------

    if dominant == "dialogue":
        # Heavy compression for old memories
        if days_gap > 30:
            return {
                "type": "dialogue",
                "text": "There was an argument about what to do next."
            }
        else:
            return {
                "type": "dialogue",
                "text": DIALOGUE_TEMPLATES[0]
            }

    # -------------------------
    # FALLBACK (description / atmosphere)
    # -------------------------

    if days_gap <= 7:
        return {
            "type": dominant,
            "text": "The scene unfolded quietly without immediate consequences."
        }

    return None
