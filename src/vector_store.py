import chromadb
from chromadb.config import Settings

def get_chroma_client(persist_dir="chroma_store"):
    return chromadb.Client(
        Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False
        )
    )

def create_chroma_collection(
    collection_name="bookmark_memory",
    persist_dir="chroma_store"
):
    client = get_chroma_client(persist_dir)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    return client, collection

def store_memories(
    collection,
    memories,
    embeddings
):
    ids = [f"mem_{i}" for i in range(len(memories))]

    metadatas = []
    documents = []

    for i, mem in enumerate(memories):
        tag = mem.split("]")[0].replace("[", "")
        metadatas.append({
            "index": i,
            "tag": tag
        })
        documents.append(mem)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
