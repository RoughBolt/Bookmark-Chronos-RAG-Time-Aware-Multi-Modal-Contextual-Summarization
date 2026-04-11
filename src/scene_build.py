def build_scenes(events, window=3):
    scenes = []
    current = []

    for e in events:
        if not current:
            current.append(e)
            continue

        if abs(e["position"] - current[-1]["position"]) <= window:
            current.append(e)
        else:
            scenes.append(current)
            current = [e]

    if current:
        scenes.append(current)

    return scenes