import ollama

def embed_memories(memories, model="nomic-embed-text"):
    assert isinstance(memories, list)
    assert all(isinstance(m, str) for m in memories)

    embeddings = []

    for mem in memories:
        response = ollama.embeddings(
            model=model,
            prompt=mem
        )
        embeddings.append(response["embedding"])

    assert len(embeddings) == len(memories)

    dim = len(embeddings[0])
    for vec in embeddings:
        assert len(vec) == dim

    return embeddings

def embed_text(text, model="nomic-embed-text"):
    response = ollama.embeddings(
        model=model,
        prompt=text
    )
    return response["embedding"]

