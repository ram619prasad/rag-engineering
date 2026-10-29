
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

### Index Structure — Baseline vs Contextual Retrieval

Both approaches use the same source sections and stable identifiers. The difference is what gets indexed.

**Baseline index (Module 02)**

```text
Section: kubernetes_hpa:s008
│
├── ID: kubernetes_hpa:s008
│
├── Indexed document:
│     ## Autoscaling during rolling update
│     Kubernetes lets you perform a rolling update...
│
├── BM25 index:
│     Tokens from original section text
│
└── Chroma index:
      Embedding(original section text)
```

**Contextual index (Module 03)**

```text
Section: kubernetes_hpa:s008
│
├── ID: kubernetes_hpa:s008
│
├── Generated context:
│     This section explains how HorizontalPodAutoscaler
│     interacts with rolling updates in Kubernetes...
│
├── Original text:
│     ## Autoscaling during rolling update
│     Kubernetes lets you perform a rolling update...
│
├── Indexed document:
│     [Generated context]
│     +
│     [Original section text]
│
├── BM25 index:
│     Tokens from context + original section
│
└── Chroma index:
      Embedding(context + original section)
```

**Comparison**

| Property | Baseline Index | Contextual Index |
|---|---|---|
| Section ID | Same | Same |
| Indexed text | Original section | Context + original section |
| BM25 input | Original tokens | Contextualized tokens |
| Embedding input | Original text | Contextualized text |
| Number of embeddings | One per section | One per section |
| Storage | Original index | Separate contextual index |
| Original source content | Unchanged | Unchanged |

**Why this matters**

A section that lacks technical names or document-level context may be difficult to retrieve independently. Adding a concise contextual prefix can improve both keyword matching and semantic similarity.

However, contextualization can also introduce redundant information or noise. We evaluate both indexes against the same benchmark rather than assuming the contextual index is better.

---

## Module 04 — Parent-Child Retrieval

**Status:** Completed and evaluated

### 1. Objective

Implement Parent-Child Retrieval to combine the precision of searching small chunks with the richer context of larger parent sections.

**Core idea: Search small, return large.**

Unlike conventional semantic retrieval, which searches and returns the same indexed chunk, Parent-Child Retrieval separates the retrieval unit from the context returned to the application.

### 2. Problem: Small vs Large Chunks

Traditional RAG faces a trade-off:

| Small chunks | Large chunks |
|---|---|
| Focused semantic representation | More complete context |
| May improve precise matching | May contain multiple concepts |
| Can lose important explanations | May dilute embedding relevance |
| Lower context size | Higher context size |

For example, a child chunk containing only an HPA scaling formula may be easy to retrieve but insufficient for explaining its variables and practical behavior.

Parent-Child Retrieval addresses this by searching small chunks and fetching their complete parent sections.

### 3. Architecture

```text
                  Engineering Knowledge Base
                          434 Sections
                               |
                               v
                      Parent Documents
                               |
                +--------------+--------------+
                |                             |
                v                             v
         Child Text Splitter              InMemoryStore
         Size: 400 chars                  Parent Documents
         Overlap: 60 chars                Key: section_id
                |                             |
                v                             |
          Child Documents                     |
                |                             |
                v                             |
           OpenAI Embeddings                  |
                |                             |
                v                             |
           Chroma Vector DB                   |
           (Child Index)                      |
                |                             |
                v                             |
          Query Similarity                    |
                |                             |
                v                             |
          Matching Children                   |
                |                             |
                v                             |
          Extract parent_id                   |
                |                             |
                +-----------------------------+
                               |
                               v
                   docstore.mget(parent_ids)
                               |
                               v
                    Deduplicate Parents
                               |
                               v
                    Return Full Sections
```

### 4. Index Structure

The parent sections are stored separately from the indexed child chunks.

**Child vector index — Chroma**

```text
Child ID: kubernetes_hpa:s008:c001
│
├── Text:
│   "Kubernetes lets you perform a rolling update
│    on a Deployment..."
│
├── Metadata:
│   parent_id: kubernetes_hpa:s008
│
└── Embedding:
    Embedding(child text)
```

**Parent document store — InMemoryStore**

```text
Key: kubernetes_hpa:s008
│
└── Document:
    ├── section_id: kubernetes_hpa:s008
    ├── document_id: kubernetes_hpa
    ├── heading_path: Autoscaling during rolling update
    └── page_content:
        Full 738-character parent section
```

The child index supports semantic similarity search.

The parent store supports direct key-based lookup without embeddings.

### 5. Real Example — Kubernetes HPA

**Query:**

> How does Kubernetes HPA manage replicas during rolling updates?

**Retrieved children:**

| Child rank | Parent ID | Matched content |
|---|---|---|
| 1 | `kubernetes_hpa:s008` | Autoscaling during rolling updates |
| 2 | `kubernetes_hpa:s008` | ReplicaSet and StatefulSet behavior |
| 3 | `kubernetes_hpa:s007` | Workload scaling stability |

**Returned parents:**

```text
1. kubernetes_hpa:s008
   Length: 738 characters
   Topic: Autoscaling during rolling update

2. kubernetes_hpa:s007
   Length: 371 characters
   Topic: Stability of workload scale
```

Three child matches produced only two parent documents because two children belonged to the same parent.

The full-corpus experiment later searched 40 child candidates and returned 20 unique parent sections for the rolling-update query. The directly relevant `kubernetes_hpa:s008` ranked first.

### 6. Implementation

**Libraries:**

- LangChain `ParentDocumentRetriever`
- `RecursiveCharacterTextSplitter`
- Chroma
- OpenAI `text-embedding-3-small`
- LangChain `InMemoryStore`

**Configuration:**

| Parameter | Value |
|---|---|
| Parent unit | Existing EKB section |
| Parent sections | 434 |
| Child chunk size | 400 characters |
| Child overlap | 60 characters |
| Child search candidates | 40 |
| Parent store | InMemoryStore |
| Child index | Persistent Chroma |
| Evaluation unit | Parent section ID |

**Implementation file:**

`src/advanced_retrieval/parent_child/parent_child_retriever.py`

**Important methods:**

- `create_parent_child_retriever()` — builds the child index and parent document store.
- `add_documents()` — splits parents, indexes children, and stores parents.
- `invoke()` — retrieves children and returns unique parent documents.

### 7. Persistence Strategy

The child vector index is persisted in Chroma.

The parent document store uses `InMemoryStore`, which does not survive application restarts.

On subsequent runs:

1. Reuse the existing Chroma child index.
2. Reload parent sections from the frozen section catalog.
3. Repopulate `InMemoryStore` using stable section IDs.
4. Resume retrieval without regenerating child embeddings.

This is appropriate for our local experiment. A production system would typically use durable document storage and stronger index-version consistency checks.

### 8. Evaluation Results

Evaluated against the same 30 development questions used in Modules 02 and 03.

| Retriever | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|
| Baseline BM25 | 0.9000 | 0.9500 | 0.8250 |
| Baseline Semantic | 0.9667 | **1.0000** | 0.9159 |
| Baseline Hybrid | 0.9167 | 0.9500 | 0.9500 |
| Contextual BM25 | **0.9833** | 0.9833 | 0.8900 |
| Contextual Semantic | 0.9667 | 0.9833 | 0.9250 |
| Contextual Hybrid | 0.9667 | 0.9833 | **0.9667** |
| **Parent-Child** | **0.9167** | **1.0000** | **0.8964** |

**Findings:**

- Parent-Child matched baseline Semantic with Recall@10 of 1.0000.
- Parent-Child Recall@5 was 0.9167, below baseline Semantic and Contextual Hybrid.
- Parent-Child MRR@10 was 0.8964, below Contextual Hybrid's 0.9667.
- Smaller child chunks did not automatically produce better retrieval rankings.
- Parent expansion returned complete original sections, which may provide more useful context for answer generation.

These metrics measure retrieval of labeled parent sections, not downstream answer correctness.

The benchmark uses automatically generated relevance labels that remain provisional.

### 9. Parent-Child vs Contextual Retrieval

| Aspect | Contextual Retrieval | Parent-Child Retrieval |
|---|---|---|
| Core idea | Add document context before indexing | Search children, return parents |
| Indexed text | Context-enriched section | Small child chunk |
| Returned text | Enriched section | Full original parent section |
| LLM preprocessing | Yes, in our implementation | No |
| Document store | Not required | Required |
| Main advantage | Improved document matching | More complete returned context |
| Primary cost | Context-generation and embedding costs | Child indexing and larger returned context |

These approaches are complementary and can be combined in future architectures.

### 10. Trade-offs and Limitations

**Advantages**

- Precise child-level similarity search.
- Complete parent context returned to the application.
- Stable parent IDs compatible with existing evaluation.
- Parent deduplication avoids returning repeated sections.
- No LLM-generated contextual prefix required.

**Limitations**

- Small child chunks may lose important query context.
- Multiple child hits can resolve to the same parent, reducing the number of unique results.
- Parent sections can vary significantly in length.
- Returning larger parents increases context-window usage.
- Persistent child indexes must remain synchronized with parent storage.
- Recall@K alone does not measure the usefulness of expanded context.

### 11. Key Learnings

1. Retrieval units and returned context units do not need to be the same.
2. Child embeddings identify relevant locations; parent IDs retrieve full context.
3. `ParentDocumentRetriever.add_documents()` manages child indexing and parent storage.
4. The parent store can use key-value lookups rather than vector similarity.
5. Multiple child matches can correspond to one unique parent.
6. Parent-Child Retrieval is not guaranteed to outperform ordinary semantic retrieval on Recall@K or MRR.
7. Retrieval quality and answer-context quality should be evaluated separately.

### 12. Conclusion

Parent-Child Retrieval achieved full labeled-section coverage at Recall@10 on our development benchmark, but did not outperform Contextual Hybrid on early ranking.

Its main architectural benefit is **decoupling precise retrieval from the amount of context returned**.

Future experiments should consider parent-context token cost and downstream answer grounding in addition to retrieval metrics.


---

## Module 05 — Hierarchical Retrieval

**Status:** Completed and evaluated

### 1. Objective

Implement top-down hierarchical retrieval using the existing Engineering Knowledge Base (EKB), comprising 30 Markdown documents and 434 extracted sections.

The goal is to explore whether selecting relevant documents before searching their sections improves retrieval quality.

**Core idea: Search broad, then narrow.**

### 2. Why Hierarchical Retrieval?

Traditional semantic retrieval searches every indexed section globally.

```text
Query
  |
  v
Search all 434 sections
  |
  v
Return top-K sections
```

Hierarchical retrieval adds a document-selection stage:

```text
Query
  |
  v
Select relevant documents
  |
  v
Search sections within selected documents
  |
  v
Return top-K sections
```

Potential benefits include improved topical focus, document-aware navigation, and reduced irrelevant search candidates.

However, an incorrect document-selection decision can exclude relevant sections before they are searched.

### 3. Architecture

```text
                  User Query
                       |
                       v
             Document-Level Retriever
                       |
                       v
            Chroma Document Index
              30 representations
                       |
                       v
             Top 3 Document IDs
                       |
                       v
               Metadata Filter
           document_id IN selected IDs
                       |
                       v
             Section-Level Retriever
                       |
                       v
           Existing Semantic Chroma
                434 sections
                       |
                       v
              Top 10 Sections
```

Two retrieval levels are used:

- **Level 0 — Document selection:** Identify relevant engineering documents using compact document representations.
- **Level 1 — Section retrieval:** Search original sections restricted to those documents.

This is a two-level, top-down retrieval baseline. It does not yet implement recursive subsection traversal, sibling expansion, or RAPTOR-style summary trees.

### 4. Document Representation

Rather than embedding entire Markdown documents, we construct one compact representation per source document.

The representation contains:

1. Document name.
2. Section headings.
3. Up to 200 characters of normalized text from each section.

The final representation is limited to 12,000 characters.

Example:

```text
Document: kubernetes hpa

Section: Stability of workload scale
When managing the scale of a group of replicas
using the HorizontalPodAutoscaler...

Section: Autoscaling during rolling update
Kubernetes lets you perform a rolling update
on a Deployment...
```

The `build_document_representations()` function groups sections by `document_id` using `defaultdict(list)` and produces one searchable record for each document.

This is deterministic preprocessing; no LLM summarization is involved.

**Limitation:** Truncation and short section excerpts can omit important content, especially from long documents.

### 5. Index Structure

The architecture reuses our existing semantic retriever with two independently stored Chroma collections.

**Document-level index**

```text
Collection: engineering_document_level

Document ID: kubernetes_hpa

Text:
  Document name
  Section headings
  Short section excerpts

Metadata:
  section_id: kubernetes_hpa
  document_id: kubernetes_hpa

Embedding:
  OpenAI text-embedding-3-small
```

The `section_id` field is reused as the document-level record identifier to maintain compatibility with our existing `SemanticRetriever` implementation.

**Section-level index**

```text
Collection: engineering_semantic_baseline

Section ID: kubernetes_hpa:s008

Text:
  Full original section content

Metadata:
  section_id: kubernetes_hpa:s008
  document_id: kubernetes_hpa
  heading_path: Autoscaling during rolling update

Embedding:
  OpenAI text-embedding-3-small
```

The document-level index selects candidate documents, while the section-level index returns the original source sections.

The section index is reused from Module 02; no additional section embeddings are generated.

### 6. Implementation

**Files:**

- `src/advanced_retrieval/hierarchical/document_retriever.py`
- `src/advanced_retrieval/hierarchical/hierarchical_retriever.py`
- `src/advanced_retrieval/evaluation/evaluate_retrievers.py`

**Libraries:**

- LangChain `RunnableLambda`
- Existing `SemanticRetriever`
- Chroma
- OpenAI `text-embedding-3-small`

**Configuration:**

| Parameter | Value |
|---|---|
| Source documents | 30 |
| Source sections | 434 |
| Document candidates (`document_k`) | 3 |
| Returned sections (`section_k`) | 10 |
| Document representation limit | 12,000 characters |
| Section excerpt limit | 200 characters |
| Evaluation unit | Original section ID |

The retrieval pipeline performs two semantic searches:

```python
documents = document_retriever.invoke(query)

document_ids = [
    document.metadata["document_id"]
    for document in documents
]

sections = section_retriever.vectorstore.similarity_search(
    query=query,
    k=10,
    filter={
        "document_id": {
            "$in": document_ids
        }
    },
)
```

Chroma applies the document-ID filter during section-level retrieval, rather than retrieving globally and filtering the results afterward.

### 7. Real Example — Kubernetes HPA

**Query:**

> How does Kubernetes HPA manage replicas during rolling updates?

**Level 0 — Retrieved documents**

| Rank | Document | Representation size |
|---|---|---:|
| 1 | `kubernetes_hpa` | 6,966 characters |
| 2 | `kubernetes_resource_management` | 8,303 characters |
| 3 | `kubernetes_pod_scheduling` | 6,190 characters |

The correct Kubernetes HPA document ranked first.

**Level 1 — Retrieved sections**

| Rank | Section ID | Topic |
|---|---|---|
| 1 | `kubernetes_hpa:s008` | Autoscaling during rolling update |
| 2 | `kubernetes_hpa:s025` | Migrating workloads to horizontal autoscaling |
| 3 | `kubernetes_hpa:s001` | How HorizontalPodAutoscaler works |
| 4 | `kubernetes_hpa:s022` | HPA support in kubectl |
| 5 | `kubernetes_hpa:s007` | Stability of workload scale |

All ten retrieved sections belonged to `kubernetes_hpa`, even though three documents were eligible.

This is expected: document selection restricts which sections may be searched, but does not require results to be distributed across selected documents.

The directly relevant rolling-update section ranked first.

### 8. Evaluation

Evaluated using the existing 30-question development benchmark and the same section-level relevance judgments used in previous modules.

| Retriever | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|
| BM25 | 0.9000 | 0.9500 | 0.8250 |
| Semantic | 0.9667 | **1.0000** | 0.9159 |
| Hybrid | 0.9167 | 0.9500 | 0.9500 |
| Contextual BM25 | **0.9833** | 0.9833 | 0.8900 |
| Contextual Semantic | 0.9667 | 0.9833 | 0.9250 |
| Contextual Hybrid | 0.9667 | 0.9833 | **0.9667** |
| Parent-Child | 0.9167 | **1.0000** | 0.8964 |
| **Hierarchical (docs=3)** | **0.9667** | **1.0000** | **0.9159** |

**Findings:**

- Hierarchical Retrieval matched baseline Semantic Retrieval on all three aggregate metrics.
- Both achieved Recall@10 of 1.0000.
- Contextual Hybrid retained the highest MRR@10 at 0.9667.
- The additional document-selection stage produced no measurable improvement in these aggregate retrieval metrics.
- Identical aggregate metrics do not necessarily mean identical per-query rankings.

These results apply to the evaluated `document_k=3` configuration. Comparisons using `document_k=1` or `document_k=5` have not yet been measured.

The benchmark's automatically generated relevance judgments remain provisional.

### 9. Parent-Child vs Hierarchical Retrieval

| Aspect | Parent-Child | Hierarchical |
|---|---|---|
| Retrieval direction | Bottom-up | Top-down |
| First search unit