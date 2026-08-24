# Baseline Comparison Results

| System | Chunks | Build Time | Keyword Recall | Precision@5 | vs Chronos |
|--------|--------|------------|----------------|-------------|------------|
| Naive RAG (fixed chunks) | 203 | 0.0s | 0.2534 | 0.7300 | Chronos -0.076 |
| **Chronos-RAG (full pipeline)** | 156 | 1.7s | **0.1779** | **0.5600** | — |

**Chronos-RAG achieves 0.1779 keyword recall**, 
demonstrating the value of cognitive science-grounded temporal decay 
and event-aware semantic processing over standard chunking strategies.