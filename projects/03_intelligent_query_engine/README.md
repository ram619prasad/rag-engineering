# Project 03 — Intelligent Query Engine

## Objective

Build an intelligent query-processing system that improves retrieval quality using query classification, rewriting, multi-query generation, decomposition, HyDE, and intelligent routing.

Measure the effectiveness of each approach against established retrieval baselines.

## Dataset

**BEIR SciFact**

Source: https://huggingface.co/datasets/Hyukkyu/beir-scifact

- Corpus: 5,183 scientific documents
- Evaluation queries: 300
- Test relevance judgments: 339

The dataset contains scientific documents, queries, and query-document relevance judgments (qrels).

Dataset files are cached locally after initial download. Original document IDs are preserved for evaluation.

## Project Tasks

- [x] Module 01 — SciFact Dataset Integration
- [x] Module 02 — Baseline Retrieval
- [x] Module 03 — Query Understanding and Classification
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

Synthetic datasets are useful for learning but insufficient for evaluating retrieval systems on realistic data.

### Implementation

- Loaded SciFact documents, queries, and qrels from Hugging Face.
- Added local dataset persistence to avoid repeated downloads.
- Validated document/query ID uniqueness.
- Validated references from qrels to documents and queries.
- Preserved benchmark identifiers.

### Results

- Dataset loaded successfully.
- Dataset validation passed.
- Benchmark data available locally for subsequent modules.

### Challenges and Limitations

Loading a dataset alone does not establish retrieval quality. We need searchable indexes and a reproducible evaluation baseline.

---

## Module 02 — Baseline Retrieval

### Problem

Different retrieval techniques have different strengths. Before implementing intelligent query transformations, we need to understand their retrieval quality, latency, and operational costs.

### Implementation

**BM25 Retrieval**

- Implemented lexical retrieval using `rank-bm25`.
- Indexed SciFact titles and abstracts.
- Preserved original document IDs.

**Chunking and Metadata**

- Initially used one chunk per scientific document.
- Introduced Pydantic models for chunks and retrieval results.
- Added deterministic chunk IDs, parent document IDs, chunk indices, source metadata, and SHA-256 content hashes.

**Semantic Retrieval**

- Used OpenAI `text-embedding-3-small` with 1,536-dimensional embeddings.
- Indexed 5,183 documents in persistent ChromaDB.
- Used cosine similarity for vector retrieval.

**Efficient Indexing**

- Estimated token usage and embedding costs before indexing.
- Generated embeddings in batches.
- Stored embeddings persistently.
- Compared content hashes to skip unchanged chunks.
- Validated embedding model and dimensions against collection configuration.
- Designed batch indexing to support resuming after partial failures.

**Hybrid Retrieval**

- Used LangChain's `EnsembleRetriever`.
- Combined BM25 and Semantic retrieval using weighted Reciprocal Rank Fusion.
- Used equal weights (0.5 / 0.5) and RRF constant `c=60`.
- Used 50 candidates per retriever for hybrid search.
- Deduplicated documents by their original document IDs.

**Evaluation**

- Used `ir-measures` for standard retrieval metrics.
- Evaluated 300 SciFact test queries against the dataset's qrels.
- Measured Recall@5, Recall@10, MRR@10, nDCG@10, and average query latency.

### Benchmark Results

| Metric | BM25 | Semantic | Hybrid |
|---|---:|---:|---:|
| Recall@5 | 0.7284 | **0.7999** | 0.7727 |
| Recall@10 | 0.7740 | **0.8549** | 0.8466 |
| MRR@10 | 0.6186 | **0.6903** | 0.6894 |
| nDCG@10 | 0.6519 | **0.7242** | 0.7228 |
| Average query latency | **5.80 ms** | 302.68 ms | 328.50 ms |

These measurements were obtained in the local development environment and are not production latency guarantees.

### Findings

1. Semantic retrieval delivered the strongest retrieval quality across all four metrics.
2. BM25 delivered substantially lower query latency.
3. Hybrid retrieval did not outperform semantic retrieval with the current equal-weight RRF configuration.
4. Retrieval quality must be measured across the full evaluation set rather than inferred from individual examples.

**Observed retrieval failure**

Query: `0-dimensional biomaterials show inductive properties.`

Known relevant document: `31715818`

- BM25: zero matching query tokens; zero relevance score.
- Semantic: relevant document appeared at rank 45.
- Hybrid: relevant document did not appear in the final top five.

This demonstrates that even a functioning retrieval pipeline can fail to prioritize relevant evidence.

### Challenges and Limitations

- Semantic retrieval introduces embedding API latency and costs.
- Hybrid retrieval introduces additional execution overhead without guaranteed relevance improvements.
- Equal RRF weights may not be optimal for SciFact.
- General-purpose embeddings may struggle with specialized scientific terminology.
- The current system lacks query understanding and intelligent strategy selection.
- Chunking is intentionally simple: one scientific abstract per chunk.
- Incremental indexing detects additions and changed text but does not yet reconcile deleted source documents.
- Query embeddings may be generated repeatedly across separate benchmark runs.

---

## Module 03 — Query Understanding & Classification

### Problem

Different queries may benefit from different retrieval strategies.

The baseline benchmark showed that semantic retrieval had the highest overall quality on SciFact, while BM25 offered substantially lower latency.

We want to understand query characteristics before deciding which retrieval method to use.

### Implementation

- Implemented `QueryClassifier` using LangChain and `gpt-4o-mini`.
- Used `ChatPromptTemplate` for classification instructions.
- Used LangChain's `with_structured_output()` with Pydantic validation.
- Classified queries into exact-match, factual, explanatory, comparison, troubleshooting, and multi-hop categories.
- Returned a suggested retrieval strategy: BM25, Semantic, or Hybrid.
- Measured end-to-end classification latency using `time.perf_counter()`.

### Results

| Query | Classification | Strategy | Latency |
|---|---|---|---:|
| Error code | Exact match | BM25 | 1608.12 ms |
| Semantic search explanation | Explanatory | Semantic | 3011.04 ms |
| BM25 vs vector search | Comparison | Hybrid | 1637.94 ms |
| Kubernetes troubleshooting | Troubleshooting | Hybrid | 1434.23 ms |
| Scientific claim | Factual | Semantic | 1531.13 ms |

**Average classification latency: 1844.49 ms.**

All five queries produced valid structured outputs with plausible classifications and retrieval recommendations.

### Findings

1. LangChain's structured-output functionality simplifies LLM classification.
2. Pydantic provides consistent classification results without manual JSON parsing.
3. The classifier produces reasonable recommendations but does not guarantee optimal retrieval.
4. LLM classification adds substantial latency compared with BM25 and semantic retrieval.
5. Query classification and retrieval strategy selection are related but distinct decisions.

### Challenges and Limitations

- LLM calls introduce latency and API costs.
- Five manually selected examples are insufficient to establish classification accuracy.
- Retrieval strategy recommendations have not yet been evaluated against query-level benchmark performance.
- An LLM call for simple exact-match queries may be unnecessary.
- A rule-based classifier or lightweight routing layer could provide a lower-latency alternative.
