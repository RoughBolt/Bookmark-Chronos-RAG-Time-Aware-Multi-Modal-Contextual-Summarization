# Stitching rules (VERY IMPORTANT)

# Rule 1: Preserve Order
# Rule 2: Collapse Weak Events
# Rule 3: Cause -> Effect LINKING
# Rule 4: Death always stands alone

def stitch_events(events, flags, mode="full"):
    # Rule: Death always stands alone
    for e in events:
        if e["type"] == "DEATH":
            return f"{e['text']}"

    texts = [e["text"] for e in events]

    if mode == "summary":
        return texts[0]

    return " ".join(texts)

def stitch_memory_timeline(memories):
    if not memories:
        return ""

    stitched = []
    buffer = []
    last_type = None

    def flush_buffer(buf, event_type):
        if not buf:
            return
        text_block = " ".join(buf)
        if event_type == "EVENT":
            stitched.append(f"Several minor events occurred during this time: {text_block}")
        elif event_type == "DISCOVERY":
            stitched.append(f"Something strange was noticed, which raised concern: {text_block}")
        elif event_type == "COMBAT":
            stitched.append(f"This escalated into a violent confrontation: {text_block}")
        elif event_type == "DEATH":
            stitched.append(f"A life was lost: {text_block}")
        elif event_type == "CRITICAL":
            stitched.append(f"A major irreversible event occurred: {text_block}")
        buf.clear()

    for m in memories:
        if m.startswith("[EVENT]"):
            curr_type = "EVENT"
            content = m.replace("[EVENT]", "").strip()
        elif m.startswith("[DISCOVERY]"):
            curr_type = "DISCOVERY"
            content = m.replace("[DISCOVERY]", "").strip()
        elif m.startswith("[COMBAT]"):
            curr_type = "COMBAT"
            content = m.replace("[COMBAT]", "").strip()
        elif m.startswith("[DEATH]"):
            curr_type = "DEATH"
            content = m.replace("[DEATH]", "").strip()
        elif m.startswith("[CRITICAL]"):
            curr_type = "CRITICAL"
            content = m.replace("[CRITICAL]", "").strip()
        else:
            curr_type = "EVENT"
            content = m.strip()

        # If the type changes, flush previous buffer
        if curr_type != last_type and buffer:
            flush_buffer(buffer, last_type)

        buffer.append(content)
        last_type = curr_type

    if buffer:
        flush_buffer(buffer, last_type)

    return " ".join(stitched)
