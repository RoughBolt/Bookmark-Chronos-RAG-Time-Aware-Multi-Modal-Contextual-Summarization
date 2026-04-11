def recall_memories(collection, query, k=5):
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    recalled = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        # Hard relevance filter
        if dist > 0.55:
            continue

        recalled.append({
            "text": doc,
            "tag": meta.get("tag"),
            "index": meta.get("index"),
            "distance": dist
        })

    # Chronological order restore
    recalled = sorted(recalled, key=lambda x: x["index"])

    return recalled
