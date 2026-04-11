from collections import Counter
from sys import flags

def dominant_event_type(scene_events):
    counts = Counter(e["type"] for e in scene_events)
    return counts.most_common(1)[0][0]

def normalize_event_type(t):
    return "".join(c for c in t.upper() if c.isalpha())

def contains_event(scene_events, event_type):
    target = normalize_event_type(event_type)
    return any(normalize_event_type(e["type"]) == target for e in scene_events)

def build_scene_flags(scene_events):
    flags = {}

    flags["critical"] = contains_event(scene_events, "DEATH")

    # Death = permanent memory marker
    if contains_event(scene_events, "DEATH"):
        flags["intensity"] = "high"
    elif contains_event(scene_events, "COMBAT"):
        flags["intensity"] = "high"
    elif contains_event(scene_events, "DISCOVERY"):
        flags["intensity"] = "medium"
    else:
        flags["intensity"] = "low"


    # Intensity modeling
    # max_importance = max(e.get("importance", 1) for e in scene_events)
    # if max_importance >= 4:
    #     flags["intensity"] = "high"
    # elif max_importance >= 3:
    #     flags["intensity"] = "medium"
    # else:
    #     flags["intensity"] = "low"

    # Memory persistence hint (future use)
    flags["long_term"] = flags["critical"] or flags["intensity"] == "high"

    return flags

def consolidate_scene(abstract_events):
    """
    Converts multiple abstract events into a single memory scene.
    """

    if not abstract_events:
        return None
    
    for e in abstract_events:
        e["type"] = e["type"].upper()

    # 1️⃣ Deduplicate while preserving narrative order
    seen = set()
    unique_events = []

    for event in abstract_events:
        key = (event["type"], event["text"])
        if key not in seen:
            seen.add(key)
            unique_events.append(event)

    # 2️⃣ Determine dominant scene type
    scene_label = dominant_event_type(unique_events)

    # 3️⃣ Attach memory flags
    flags = build_scene_flags(unique_events)

    return {
        "scene_label": scene_label,
        "events": unique_events,
        "flags": flags 
    }