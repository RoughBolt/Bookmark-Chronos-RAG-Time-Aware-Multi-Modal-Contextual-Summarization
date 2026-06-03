import chromadb
from src.knowledge.characters import extract_characters


def get_chroma_client(persist_dir="chroma_store"):
    # PersistentClient saves to disk; Client() is strictly in-memory.
    return chromadb.PersistentClient(path=persist_dir)


def create_chroma_collection(
    collection_name="bookmark_memory",
    persist_dir="chroma_store"
):
    client = get_chroma_client(persist_dir)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:search_ef": 50,       # Query-time expansion factor (default=10). Higher = better recall.
            "hnsw:construction_ef": 50, # Optional: just adding construction_ef to be safe
            "hnsw:M": 16,        # Graph connectivity at index time (default=16). Better graph quality.
        }
    )

    return client, collection


def clear_collection(client, collection_name="bookmark_memory"):
    """
    Deletes and recreates the collection. Used before a full re-index
    to avoid duplicate memory entries.
    """
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # Collection may not exist yet on first run
    return client.create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:search_ef": 50,
            "hnsw:construction_ef": 50,
            "hnsw:M": 16,
        }
    )


def store_memories(collection, memories, embeddings):
    """
    Stores memory strings + their embeddings into ChromaDB.
    Metadata includes: index, tag, and characters (for G-RAG neighborhood expansion).
    """
    ids = [f"mem_{i}" for i in range(len(memories))]
    metadatas = []
    documents = []

    for i, mem in enumerate(memories):
        tag = mem.split("]")[0].replace("[", "") if "]" in mem else "EVENT"

        # Extract characters for G-RAG graph traversal at query time
        chars = extract_characters(mem)
        characters_str = ",".join(chars) if chars else ""

        metadatas.append({
            "index": i,
            "tag": tag,
            "characters": characters_str
        })
        documents.append(mem)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
