# src/event_chain.py

PRIORITY = {
    "death": 4,
    "combat": 3,
    "discovery": 2,
    "dialogue": 1,
    "description": 0,
    "atmosphere": 0
}

def chain_events(events, window=3):
    chains = []
    current_chain = []

    for e in events:
        if not current_chain:
            current_chain.append(e)
            continue

        last = current_chain[-1]

        # If events are close in text → same chain
        if abs(e["position"] - last["position"]) <= window:
            current_chain.append(e)
        else:
            chains.append(current_chain)
            current_chain = [e]

    if current_chain:
        chains.append(current_chain)

    return chains


def compress_chain(chain):
    # Pick the most important event in the chain
    return max(chain, key=lambda e: PRIORITY.get(e["type"], 0))


def build_narrative(events):
    narrative = []
    last_type = None

    valid_flow = {
    None: ["dialogue", "discovery", "atmosphere"],

    "dialogue": ["dialogue", "discovery", "combat"],

    "atmosphere": ["dialogue", "discovery"],

    "discovery": ["dialogue", "combat", "discovery"],

    "combat": ["dialogue", "combat", "death"],

    "death": []
}


    for e in events:
        if last_type is None:
            narrative.append(e)
            last_type = e["type"]
            continue

        if e["type"] in valid_flow.get(last_type, []):
            narrative.append(e)
            last_type = e["type"]

    return narrative