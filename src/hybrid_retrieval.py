import re
from rank_bm25 import BM25Okapi

def _tokenize(text: str) -> list:
    """Simple lowercase word tokenizer for BM25."""
    return re.findall(r'\w+', text.lower())

def hybrid_search(
    query: str, 
    collection, 
    documents: list, 
    embed_fn, 
    k: int = 5, 
    dense_weight: float = 1.0, 
    sparse_weight: float = 1.0, 
    rrf_k: int = 60
) -> list:
    """
    Performs Hybrid Search (Dense + BM25) and combines using Reciprocal Rank Fusion (RRF).
    
    Args:
        query: The search query
        collection: The ChromaDB collection (for dense retrieval)
        documents: The list of raw text memories (for BM25 indexing)
        embed_fn: The embedding function for the query
        k: The number of final documents to return
        dense_weight: Multiplier for dense RRF score
        sparse_weight: Multiplier for sparse RRF score
        rrf_k: The RRF constant
    
    Returns:
        List of the top K retrieved documents as strings
    """
    if not documents:
        return []
    
    # 1. Dense Search (Chroma)
    q_vec = embed_fn(query)
    dense_results = collection.query(
        query_embeddings=[q_vec],
        n_results=min(20, len(documents)),
        include=["documents", "distances"]
    )
    dense_docs = dense_results["documents"][0] if dense_results["documents"] else []
    
    # 2. Sparse Search (BM25)
    tokenized_corpus = [_tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = _tokenize(query)
    
    # Get scores for all documents and sort to find top matches
    sparse_scores = bm25.get_scores(tokenized_query)
    sparse_scored_docs = list(zip(documents, sparse_scores))
    sparse_scored_docs.sort(key=lambda x: x[1], reverse=True)
    sparse_docs = [doc for doc, score in sparse_scored_docs[:min(20, len(documents))]]
    
    # 3. Reciprocal Rank Fusion
    rrf_scores = {}
    
    # Assign RRF from Dense
    for rank, doc in enumerate(dense_docs):
        if doc not in rrf_scores:
            rrf_scores[doc] = 0.0
        rrf_scores[doc] += dense_weight * (1.0 / (rrf_k + rank + 1))
        
    # Assign RRF from Sparse
    for rank, doc in enumerate(sparse_docs):
        if doc not in rrf_scores:
            rrf_scores[doc] = 0.0
        rrf_scores[doc] += sparse_weight * (1.0 / (rrf_k + rank + 1))
        
    # 4. Sort and return Top K
    fused_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in fused_docs[:k]]
