# Project 03 — Intelligent Query Engine

## Objective

Build an intelligent query-processing system that improves retrieval quality through query rewriting, multi-query retrieval, query decomposition, HyDE, and intelligent routing.

Evaluate each approach against a baseline using real-world data.

## Dataset

**BEIR SciFact**

Source: https://huggingface.co/datasets/Hyukkyu/beir-scifact

The dataset contains scientific documents, queries, and relevance judgments (qrels).

- Corpus: Documents used for retrieval
- Queries: Search queries used for evaluation
- Qrels: Ground-truth query-document relevance judgments

The dataset is stored locally after initial loading to support repeatable, offline experiments.

## Project Tasks

- [x] Module 01 — SciFact Dataset Integration
- [ ] Module 02 — Baseline Retrieval
- [ ] Module 03 — Query Understanding and Classification
- [ ] Module 04 — Query Rewriting
- [ ] Module 05 — Multi-Query Retrieval
- [ ] Module 06 — Query Decomposition
- [ ] Module 07 — HyDE
- [ ] Module 08 — Intelligent Query Routing
- [ ] Module 09 — Evaluation and Comparison
- [ ] Module 10 — Documentation and Finalization

---

## Module 01 — SciFact Dataset Integration

### Problem

Synthetic datasets are useful for learning retrieval concepts but are insufficient for evaluating real-world retrieval quality.

We need a real benchmark with documents, queries, and relevance judgments to measure retrieval improvements objectively.

### Implementation

Implemented `dataset.py` to:

1. Load SciFact corpus, queries, and relevance judgments from Hugging Face.
2. Save the downloaded dataset locally and reuse it on subsequent executions.
3. Validate document and query ID uniqueness.
4. Verify that relevance judgments reference existing documents and queries.
5. Report dataset statistics and relevance-label distribution.

### Results

- Dataset loading: Successful
- Dataset validation: PASSED
- Corpus, queries, and qrels are available for subsequent modules.

### Challenges and Limitations

- Hugging Face requests may require authentication for higher rate limits.
- Repeated remote loading introduces unnecessary network overhead.
- Local dataset caching helps avoid repeated downloads.
- The dataset is loaded and validated, but no search indexes or retrieval benchmarks have been built yet.

### Next Module — Baseline Retrieval

Implement BM25, semantic, and hybrid retrieval against the SciFact corpus.

Evaluate baseline performance using Recall@K, MRR, and nDCG@K.

These scores will establish a reference point for measuring improvements from query rewriting, multi-query retrieval, HyDE, and intelligent routing.