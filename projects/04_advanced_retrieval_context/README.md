
# Project 04 — Advanced Retrieval & Context Engineering

## Objective

Build and evaluate advanced RAG retrieval strategies using a fixed
Engineering Knowledge Base (EKB).

The project compares retrieval quality across:

1. BM25, Semantic and Hybrid Retrieval
2. Contextual Retrieval
3. Parent-Child Retrieval
4. Hierarchical Retrieval
5. Advanced and Adaptive Chunking
6. Late Chunking
7. Long-Context Retrieval

All experiments use the same underlying document corpus and
evaluation questions to enable meaningful comparisons.

## Progress

| Module | Topic | Status |
|---|---|---|
| 01 | Engineering Dataset & Evaluation Benchmark | Completed (provisional labels) |
| 02 | BM25, Semantic & Hybrid Retrieval | Next |
| 03 | Contextual Retrieval | Pending |
| 04 | Parent-Child Retrieval | Pending |
| 05 | Hierarchical Retrieval | Pending |
| 06 | Advanced / Adaptive Chunking | Pending |
| 07 | Late Chunking | Pending |
| 08 | Long-Context Retrieval | Pending |
| 09 | Evaluation & Architecture Review | Pending |

## Module 01 — Engineering Dataset & Evaluation Benchmark

### Dataset

Created an Engineering Knowledge Base containing 30 real-world
technical Markdown documents across seven domains.

| Domain | Documents |
|---|---:|
| Kubernetes | 6 |
| Docker | 2 |
| Prometheus | 4 |
| Distributed Systems | 5 |
| Microservices | 4 |
| PostgreSQL | 5 |
| Caching | 4 |
| **Total** | **30** |

### Document Processing

Implemented:

- Markdown document downloading using HTTPX
- Document loading using LangChain TextLoader
- Section extraction using MarkdownHeaderTextSplitter
- Hierarchical heading metadata preservation
- Stable document and section identifiers
- Corpus inspection and statistics
- Source manifest with SHA-256 hashes

Results:

- Documents: 30
- Extracted sections: 434
- Average sections per document: approximately 14.5

### Dataset Artifacts

All dataset artifacts are stored under:

`datasets/engineering_kb/`

Important files:

| File | Purpose |
|---|---|
| `documents/` | Source Markdown documents |
| `sources.json` | Frozen source metadata and hashes |
| `section_catalog.json` | Extracted sections and stable IDs |
| `benchmark_candidates.json` | Generated evaluation questions |
| `benchmark_dev_generated.json` | Development questions |
| `benchmark_test_generated.json` | Held-out test questions |

### Evaluation Benchmark

Generated 120 retrieval questions.

| Question Type | Count |
|---|---:|
| Factual | 50 |
| Comparison | 30 |
| Multi-section | 25 |
| Cross-document | 15 |
| **Total** | **120** |

Development/test split:

- Development: 30 questions
- Test: 90 questions
- Random seed: 42
- Both sets use the same 30-document corpus

### Validation Results

Structural validation PASSED.

Verified:

- Expected question counts and category distribution
- Unique question identifiers
- No exact duplicate questions
- Referenced section IDs exist in the catalog
- Cross-document questions reference multiple documents
- Reproducible development/test split

**Limitation:** Relevance labels were generated automatically.
Structural validation does not establish that every referenced
section is necessary or sufficient to answer its question.
Evaluation scores should therefore be treated as provisional
until label quality is reviewed.

### Key Learnings

1. Preserve document hierarchy before experimenting with chunking.
2. Stable section IDs make retrieval experiments comparable.
3. Freeze the corpus to avoid changes between experiments.
4. Development questions support retrieval configuration tuning.
5. Held-out questions help assess generalization.
6. Automatically generated relevance labels require quality checks.

---

