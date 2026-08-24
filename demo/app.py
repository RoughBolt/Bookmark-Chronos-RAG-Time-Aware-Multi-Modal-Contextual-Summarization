import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import streamlit as st
import sys
import time
import ollama
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import find_chapters, extract_text_upto_chapter
from src.preprocessing import clean_text
from src.bookmark import load_bookmark
from experiments.baseline_comparison import chunk_naive, build_chronos_memories
from src.embeddings import embed_memories, embed_text
from src.vector_store import create_chroma_collection

st.set_page_config(page_title="Chronos-RAG Demo", layout="wide")

st.title("Chronos-RAG: Time-Aware Narrative RAG")
st.markdown("""
This interactive demo compares **Vanilla RAG** (naive chunking) with **Chronos-RAG** (our time-aware, multi-scale memory pipeline with semantic deduplication and knowledge graph augmentation).
""")

@st.cache_resource
def prepare_data():
    bm = load_bookmark()
    chapters = find_chapters('data/book.txt')
    text = extract_text_upto_chapter('data/book.txt', chapters, bm['pov'], bm['occurrence'])
    text = clean_text(text)
    
    # 1. Build Naive RAG (Vanilla)
    naive_chunks = chunk_naive(text)
    naive_embeddings = embed_memories(naive_chunks)
    import chromadb
    client_n = chromadb.Client()
    try: client_n.delete_collection("demo_naive")
    except: pass
    collection_n = client_n.create_collection("demo_naive", metadata={"hnsw:space": "cosine"})
    collection_n.add(embeddings=naive_embeddings, documents=naive_chunks, ids=[f"n_{i}" for i in range(len(naive_chunks))])
    
    # 2. Build Chronos-RAG
    days_gap = (time.time() - os.path.getmtime("data/bookmark.json")) / (24 * 3600)
    days_gap = max(7, days_gap)  # default to 7 days if file was recently touched

    cr_chunks = build_chronos_memories(text, int(days_gap))  # positional, not keyword
    cr_embeddings = embed_memories(cr_chunks)
    client_c2 = chromadb.Client()
    try: client_c2.delete_collection("demo_chronos")
    except: pass
    collection_c = client_c2.create_collection("demo_chronos", metadata={"hnsw:space": "cosine"})
    collection_c.add(embeddings=cr_embeddings, documents=cr_chunks, ids=[f"c_{i}" for i in range(len(cr_chunks))])
    
    return collection_n, collection_c, len(naive_chunks), len(cr_chunks), bm

with st.spinner("Initializing collections and embedding book text (this may take a minute on first run)..."):
    col_n, col_c, n_count, c_count, bm = prepare_data()

st.success(f"Data loaded up to bookmark **{bm['pov']} {bm['occurrence']}**. Vanilla RAG created {n_count} chunks. Chronos-RAG extracted {c_count} episodic memories.")

query = st.text_input("Ask a question about the story:")

def generate_answer(query, context):
    prompt = f"Use the following story context to answer the question. If you don't know the answer based on the context, say so.\n\nContext:\n{'- ' + chr(10).join(context)}\n\nQuestion: {query}\nAnswer:"
    response = ollama.generate(model="qwen2:0.5b", prompt=prompt)
    return response["response"]

if query:
    q_vec = embed_text(query)
    
    # Query Vanilla
    res_n = col_n.query(query_embeddings=[q_vec], n_results=5, include=["documents"])
    ctx_n = res_n["documents"][0] if res_n["documents"] else []
    
    # Query Chronos
    res_c = col_c.query(query_embeddings=[q_vec], n_results=5, include=["documents"])
    ctx_c = res_c["documents"][0] if res_c["documents"] else []
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Vanilla RAG")
        with st.spinner("Generating answer..."):
            ans_n = generate_answer(query, ctx_n)
        st.write(ans_n)
        with st.expander("Retrieved Context (Top 5)"):
            for i, c in enumerate(ctx_n):
                st.info(c)
                
    with col2:
        st.subheader("Chronos-RAG")
        with st.spinner("Generating answer..."):
            ans_c = generate_answer(query, ctx_c)
        st.write(ans_c)
        with st.expander("Retrieved Context (Top 5)"):
            for i, c in enumerate(ctx_c):
                st.success(c)
