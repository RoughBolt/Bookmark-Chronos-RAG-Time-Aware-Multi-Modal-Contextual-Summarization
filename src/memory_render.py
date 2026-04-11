from src.narrative_stitch import stitch_events

def render_detailed(events):
    return " ".join(e["text"] for e in events)

def render_scene_summary(label, events):
    if label == "COMBAT":
        return "A confrontation broke out, ending in bloodshed."
    if label == "DISCOVERY":
        return "Something unnatural was discovered in the forest."
    if label == "DEATH":
        return "Someone was found dead."
    return "An important scene unfolded."

def render_outcome_summary(flags):
    if flags.get("critical"):
        return "The encounter ended in death."
    if flags.get("intensity") == "high":
        return "A violent encounter took place."
    return "An event of note occurred."

def render_critical_only(flags):
    if flags.get("critical"):
        return "Someone died."
    return "Something happened, but the details are unclear."

def render_short_term(scene):
    """
    Detailed but compressed recall for recent memory (<= 3 days)
    """
    events = scene["events"]

    lines = []
    for e in events:
        if e["type"] in {"DISCOVERY", "COMBAT", "DEATH"}:
            lines.append(e["text"])

    return " ".join(lines)

def render_memory_scene(scene, days_gap):
    events = scene["events"]
    flags = scene["flags"]

    # Very recent → full narrative
    if days_gap <= 3:
        return render_short_term(scene)

    # Recent → summarized narrative
    if days_gap <= 14:
        return stitch_events(events, flags, mode="summary")

    # Mid-term → outcome only
    if days_gap <= 90:
        return render_outcome_summary(flags)

    # Long-term → critical only
    return render_critical_only(flags)