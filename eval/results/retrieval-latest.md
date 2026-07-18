Real retrieval run using `BAAI/bge-small-en-v1.5`; ranking cutoff k=5.

| Configuration | Precision@5 | Recall@5 | MRR | Groundedness | Status |
|---|---:|---:|---:|---:|---|
| Dense only | 22.14% | 89.29% | 0.8810 | 96.43% | completed |
| Dense + FTS (RRF) | 22.86% | 92.86% | 0.9315 | 92.86% | completed |
| Hybrid + reranker | 23.57% | 94.64% | 0.9643 | 96.43% | completed |

Groundedness is `n/a` unless the optional provider-backed judge was run. An unavailable reranker is reported explicitly and is never relabeled as a reranked run.
