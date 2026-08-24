# Ablation Study Results

| Config | Description | Memories | Build Time | Keyword Recall | Precision@5 | Δ Recall |
|--------|-------------|----------|------------|----------------|-------------|----------|
| C0 | Vanilla RAG | 1164 | 0.0s | **0.1404** | 0.7055 | +0.000 |
| C1 | + Semantic Chunking | 387 | 0.1s | **0.2080** | 0.7891 | +0.068 |
| C2 | + Event Classification | 387 | 1.5s | **0.2080** | 0.7891 | +0.068 |
| C3 | + Single-Scale Decay (hand-tuned) | 387 | 0.2s | **0.2080** | 0.7891 | +0.068 |
| C4 | + Multi-Scale Decay (Atkinson-Shiffrin) | 387 | 0.2s | **0.2080** | 0.7891 | +0.068 |
| C5 | + Knowledge Graph (G-RAG) | 387 | 0.2s | **0.2029** | 0.7764 | +0.062 |
| C6 | + Semantic Deduplication | 387 | 29.7s | **0.2029** | 0.7764 | +0.062 |
| C7 | Full Chronos-RAG | 387 | 0.2s | **0.2029** | 0.7764 | +0.062 |
| C8 | + Hybrid Search (BM25 + RRF) | 387 | 0.2s | **0.2111** | 0.7964 | +0.071 |

**Best configuration:** C8 (+ Hybrid Search (BM25 + RRF))
**Improvement over Vanilla RAG:** +50.4% keyword recall
