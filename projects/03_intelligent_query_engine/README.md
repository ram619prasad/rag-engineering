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

---

## Module 04 — Query Rewriting

### Problem

User queries may be vague, conversational, or poorly aligned with terminology in source documents.

Query rewriting aims to improve retrieval by transforming queries into clearer, search-oriented expressions while preserving their original intent.

### Implementation

- Implemented `QueryRewriter` using LangChain.
- Used `ChatPromptTemplate`, `ChatOpenAI` (`gpt-4o-mini`), and `StrOutputParser`.
- Added input validation and a fallback for empty model responses.
- Added concurrent rewriting using LangChain's built-in `.batch()` API with `max_concurrency=5`.
- Reused the existing SemanticRetriever, ChromaDB index, and `ir-measures` evaluation infrastructure.
- Compared original and rewritten queries across all 300 SciFact test queries.

### Evaluation Results

| Metric | Original | Rewritten |
|---|---:|---:|
| Recall@5 | 0.7999 | 0.7891 |
| Recall@10 | 0.8549 | 0.8609 |
| MRR@10 | 0.6903 | 0.6921 |
| nDCG@10 | 0.7242 | 0.7277 |

### Latency Results

- Batch rewriting time for 300 queries: 20.15 seconds
- Amortized rewriting time/query: 67.15 ms
- Original semantic retrieval/query: 334.64 ms
- Rewritten semantic retrieval/query: 349.49 ms
- Amortized rewriting + retrieval/query: 416.65 ms
- Added amortized latency: approximately 24.5%

The amortized rewriting figure measures batch throughput, not individual LLM request latency.

### Findings

1. Rewriting improved Recall@10 by 0.60 percentage points but reduced Recall@5 by 1.08 percentage points.
2. MRR@10 and nDCG@10 showed marginal improvements.
3. Rewriting did not demonstrate a clear overall retrieval-quality advantage.
4. LangChain batching enabled concurrent LLM requests without custom concurrency implementation.
5. Rewriting introduced additional API calls, operational cost, and latency.
6. Scientific claims that are already precise may gain little from rewriting.

### Challenges and Limitations

- Rewriting may alter the original query intent or introduce assumptions.
- Quality improvements were small and have not been tested for statistical significance.
- No query-level failure analysis has yet established which classes of queries benefit most.
- Batched throughput does not directly represent real-time single-query latency.
- Semantic retrieval remains sequential in this experiment.
- The experiment did not compare alternative rewrite prompts or models.

### Architectural Decision

Do not enable unconditional query rewriting as the default retrieval strategy based on these results.

Retain the original semantic retrieval baseline and investigate selective rewriting when query characteristics justify it.

---

## Module 05 — Multi-Query Retrieval

### Problem

Single-query retrieval may fail to discover relevant documents when the user's query does not align well with terminology in the document collection.

Multi-query retrieval addresses this by generating alternative query formulations and retrieving documents using each variation.

The objective is to improve evidence coverage through multiple search perspectives.

### Implementation

- Used LangChain's `MultiQueryRetriever` from `langchain-classic`.
- Integrated the existing `SemanticRetriever` through a lightweight LangChain `BaseRetriever` adapter.
- Used `gpt-4o-mini` to generate three alternative queries.
- Enabled `include_original=True` to retain the original query.
- Retrieved five documents per query formulation.
- Reused the existing OpenAI embeddings and persistent ChromaDB index.
- Used LangChain's built-in document deduplication.
- Evaluated candidate recall using `ir-measures` and SciFact relevance judgments.
- Used a reproducible random sample of ten queries with seed 42.

### Architecture

```text
Original Query
      |
      v
LangChain MultiQueryRetriever
      |
      +-- Original Query
      +-- Generated Query 1
      +-- Generated Query 2
      +-- Generated Query 3
      |
      v
Semantic Retrieval per Query
      |
      v
Combine and Deduplicate Documents
      |
      v
Unique Candidate Document Pool
```

### Evaluation Results

| Metric | Semantic Top-20 | Multi-Query |
|---|---:|---:|
| Candidate Recall | 0.8000 | 0.7000 |
| Average Unique Candidates | 20 | 7.70 |
| Average Latency | 323.17 ms | 3684.23 ms |

**Evaluation scope:** Ten randomly selected SciFact test queries, with a fixed random seed of 42.

The semantic baseline retrieved twenty documents per query. Multi-query retrieved five per formulation, followed by deduplication.

This is an exploratory candidate-coverage comparison rather than an equal-unique-candidate-budget benchmark.

### Observations

1. Multi-query generation worked correctly using LangChain's built-in functionality.
2. Alternative queries frequently expressed similar meanings rather than exploring substantially different retrieval perspectives.
3. Retrieved documents overlapped considerably, reducing the size of the unique candidate pool.
4. Multi-query candidate recall was lower than the single-query semantic baseline in this sample.
5. Multi-query introduced substantial additional latency from LLM query generation and repeated semantic retrieval.
6. Generating more queries does not guarantee retrieving more unique relevant documents.

### Dataset Suitability

SciFact primarily contains scientific claims and focused factual retrieval tasks.

Many queries are already precise, reducing the opportunity for alternative formulations to recover additional information.

Multi-query retrieval may be more beneficial for conversational, ambiguous, broad, or multi-faceted questions that require complementary evidence.

The results should not be generalized to other datasets or use cases without further experiments.

### Challenges and Limitations

- Evaluation covered only ten queries.
- Semantic and multi-query configurations used different unique candidate budgets.
- Generated queries were not necessarily diverse.
- The deduplicated document union did not provide a meaningful global relevance ranking.
- Candidate recall was evaluated, but MRR and nDCG were not calculated for the multi-query union.
- LLM generation and repeated retrieval increased API overhead and latency.
- No query-level analysis was conducted to determine which query characteristics benefit most.

---

## Module 06 — Query Decomposition

### Problem

Complex user questions may contain multiple distinct information needs that a single retrieval query does not fully address.

Query decomposition breaks a complex question into smaller, focused subquestions so relevant evidence can be retrieved for each part.

Unlike multi-query retrieval, the objective is not to generate paraphrases but to identify distinct subproblems.

### Implementation

- Implemented `QueryDecomposer` using LangChain's `ChatPromptTemplate` and `ChatOpenAI`.
- Used Pydantic structured output to represent decomposed questions.
- Configured decomposition to produce distinct subquestions and avoid unnecessarily splitting simple queries.
- Reused the existing semantic retriever and persistent ChromaDB index.
- Retrieved the top three documents for each subquestion.
- Recorded document IDs and measured overlap between retrieved evidence sets.

### Experiment

**Original question:**

How do inflammatory biomarkers predict COPD exacerbations, and how do exacerbations affect patient survival?

**Subquestion 1:**

How do inflammatory biomarkers predict COPD exacerbations?

Retrieved document IDs:
- `21287352`
- `40935722`
- `29564505`

**Subquestion 2:**

What is the impact of COPD exacerbations on patient survival?

Retrieved document IDs:
- `19464037`
- `18988265`
- `2032877`

### Results

| Measurement | Result |
|---|---:|
| Generated subquestions | 2 |
| Documents retrieved per subquestion | 3 |
| Total retrieval results | 6 |
| Unique documents | 6 |
| Duplicate occurrences | 0 |

### Findings

1. The LLM successfully separated a multi-part question into two distinct information needs.
2. Each subquestion retrieved a different set of scientific documents.
3. No document overlap was observed in the six retrieved results.
4. The experiment demonstrated candidate diversity but did not establish improved retrieval relevance or answer correctness.
5. Query decomposition can be useful when a question requires evidence addressing multiple distinct topics.

### Challenges and Limitations

- Decomposition introduces an additional LLM call and multiple retrieval operations.
- Generated subquestions may omit important constraints or change the original intent.
- Candidate diversity alone does not establish evidence quality.
- No reference answer or multi-hop relevance judgments were used for this experiment.
- No final answer synthesis or quantitative improvement comparison was performed.
- Decomposition may be unnecessary or harmful for simple questions.

### Architectural Decision

Use query decomposition selectively for complex, multi-part questions rather than applying it universally.

Further multi-hop evaluation will be addressed in the Agentic AI learning track using a benchmark selected for that purpose.

---

## Module 07 — HyDE (Hypothetical Document Embeddings)

### Problem

Short or abstract queries may use different terminology from relevant source documents, resulting in poor semantic retrieval.

HyDE attempts to bridge this vocabulary gap by generating a hypothetical relevant document and using its embedding to retrieve real evidence.

### Implementation

- Implemented `HyDERetriever` using LangChain's `ChatPromptTemplate`, `ChatOpenAI`, and `StrOutputParser`.
- Used `gpt-4o-mini` to generate hypothetical scientific passages.
- Reused the existing SemanticRetriever, OpenAI embeddings, and persistent ChromaDB index.
- Used hypothetical passages only for retrieval, not as verified evidence.
- Evaluated retrieval using `ir-measures` against SciFact qrels.
- Compared Semantic and HyDE on ten reproducibly sampled queries with random seed 42.

### Notable Successful Example

**Original query:** `0-dimensional biomaterials show inductive properties.`

**Known relevant document:** `31715818` — *New opportunities: the use of nanotechnologies to manipulate and track stem cells.*

- Ordinary Semantic Retrieval: relevant document absent from top five; previously observed at rank 45.
- HyDE Retrieval: relevant document ranked first.

The generated hypothetical passage introduced concepts related to nanoparticles, stem cells, and tissue regeneration, which may explain its improved alignment with the relevant document.

This is a single-query success, not evidence of consistent improvement.

### Benchmark Results

| Metric | Semantic | HyDE |
|---|---:|---:|
| Recall@5 | 0.7000 | 0.6000 |
| Recall@10 | 0.8000 | 0.7000 |
| MRR@10 | 0.6343 | 0.5625 |
| nDCG@10 | 0.6720 | 0.5946 |
| Average latency | 282.62 ms | 2828.53 ms |

**Evaluation scope:** Ten randomly sampled SciFact test queries using seed 42.

### Findings

1. HyDE can recover relevant documents missed by direct semantic search.
2. HyDE performed worse across all four retrieval metrics in the sampled benchmark.
3. Hypothetical documents may introduce terminology that helps retrieval, but may also cause semantic drift.
4. HyDE introduced approximately 10 times the retrieval latency in this experiment.
5. Generated hypothetical content must not be treated as factual evidence.

### Limitations

- Only ten queries were evaluated quantitatively.
- Generated passages were not systematically analyzed for semantic drift.
- Results may vary with model choice, prompt, and embedding model.
- The experiment did not establish which query categories reliably benefit from HyDE.
- HyDE incurs additional LLM generation costs.

### Architectural Decision

Keep HyDE as an optional, specialized retrieval strategy rather than enabling it by default.

The existing Semantic Retriever remains the preferred baseline until further evidence demonstrates a reliable advantage.
