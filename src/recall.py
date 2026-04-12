# Cognitive Control Unit 👇🏼

def get_recall_parameters(days_gap, intensity="auto"):
    """
    Returns time-aware recall parameters: target k, dist threshold, and mode.
    Mode 'last_event' forces filtering the response down to just the single most recent relevant event.
    """
    if intensity == "auto":
        if days_gap <= 1:
            intensity = "low"
        elif days_gap <= 14:
            intensity = "medium"
        else:
            intensity = "high"

    if intensity == "low":
        return {"k": 3, "max_distance": 0.45, "mode": "last_event"}
    elif intensity == "medium":
        return {"k": 7, "max_distance": 0.55, "mode": "few_events"}
    else:  # high
        return {"k": 15, "max_distance": 0.65, "mode": "full_narrative"}

# Cognitive Control Unit 👆🏼

def recall_memories(collection, query_embedding, days_gap=7, intensity="auto"):
    
    params = get_recall_parameters(days_gap, intensity)
    target_k = params["k"]
    base_max_dist = params["max_distance"]
    mode = params["mode"]

    current_dist_threshold = base_max_dist
    recalled = []

    # Dynamic Distance Back-off (Edge Case #5 Fix)
    while current_dist_threshold <= 0.8:
        results = collection.query(
            query_embeddings=[query_embedding],  # Pass the vector array directly!
            n_results=target_k,
            include=["documents", "metadatas", "distances"]
        )

        recalled = []

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            # Hard relevance filter
            if dist > current_dist_threshold:
                continue

            raw_text = doc.split("]", 1)[-1].strip() if "]" in doc else doc

            # Narrative Meaningfulness: Skip very short segments
            if len(raw_text.split()) < 4:
                continue

            # Dialogue Filter: Skip quote-heavy generic events
            if meta.get("tag") == "EVENT":
                quote_count = doc.count('"') + doc.count("'")
                if quote_count >= 2:
                    # If it's mostly dialogue, require a much stricter distance
                    if dist > 0.45:
                        continue

            recalled.append({
                "text": doc,
                "tag": meta.get("tag"),
                "index": meta.get("index"),
                "distance": dist
            })

        if recalled:
            break
            
        # Relax threshold slightly if 0 vectors returned
        current_dist_threshold += 0.02
        current_dist_threshold = round(current_dist_threshold, 3)

    if not recalled:
        return []

    # Chronological order restore
    recalled = sorted(recalled, key=lambda x: x["index"])
    
    # Apply Time-Aware intensity limits
    if mode == "last_event":
        # Only keep the absolute most recent event among the matching subset
        recalled = [recalled[-1]]

    return recalled
