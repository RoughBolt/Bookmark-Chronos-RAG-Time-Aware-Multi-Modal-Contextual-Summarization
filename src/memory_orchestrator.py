from src.scene_consolidation import consolidate_scene
from src.memory_render import render_memory_scene

def is_scene_recallable(scene, days_gap):
    flags = scene["flags"]

    if days_gap <= 3:
        return True

    if days_gap <= 14:
        return True  # recall everything, but render shallow

    if days_gap <= 90:
        return flags.get("critical", False)

    return flags.get("long_term", False)

def generate_memory_recall(scenes):
    memories = []

    for scene in scenes:
        for event in scene:
            text = event.get("text", "").strip()
            etype = event.get("type", "").upper()

            if not text:
                continue

            if etype == "DISCOVERY":
                memories.append(f"[DISCOVERY] {text}")
            elif etype == "COMBAT":
                memories.append(f"[COMBAT] {text}")
            elif etype == "DEATH":
                memories.append(f"[DEATH] {text}")
            else:
                memories.append(f"[EVENT] {text}")

    # Return a list of memories, not a joined string
    return memories


# def generate_memory_recall(scenes, days_gap):
#     memories = []

#     for scene in scenes:
#         for event in scene:
#             text = event.get("text", "").strip()
#             etype = event.get("type", "").upper()

#             if not text:
#                 continue

#             # Use actual text; optional tagging
#             if etype in ["DISCOVERY", "COMBAT", "DEATH"]:
#                 memories.append(text)
#             else:
#                 # Include dialogues/events as-is
#                 memories.append(text)

#     return memories
