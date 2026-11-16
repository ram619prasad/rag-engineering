# Project 01 — End-to-End Research RAG

Build an evidence-grounded RAG system that answers questions about scientific research papers, provides citations, and abstains when sufficient evidence is unavailable.

Part of the **16-Project RAG Architect Mastery** learning series.

## 1. Tech Stack

- **Python + uv** — Project and dependency management
- **QASPER (Hugging Face)** — Research paper QA dataset
- **LangChain** — Document processing and LLM integration
- **Chroma** — Persistent vector database
- **OpenAI** — Embeddings and answer generation
- **Pydantic** — Structured LLM output
- **ir-measures** — Retrieval evaluation
- **DeepEval** — Generation evaluation

## 2. Architecture

The architecture has three main components:

**Offline indexing**

QASPER → Paragraph extraction → OpenAI embeddings → Chroma

**Online question answering**

Question + Paper ID → Semantic retrieval → Grounded LLM → Answer + Citations

**Evaluation**

Gold evidence + Retrieval results → ir-measures

Generated answers + Retrieved context → DeepEval

The Excalidraw architecture diagram is available in the `architecture/` directory.

## 3. Dataset

**Dataset:** [allenai/qasper](https://huggingface.co/datasets/allenai/qasper)

The QASPER training split contains 888 research papers.

Each paper includes:

- Paper ID, title, and abstract
- Sections and paragraphs
- Questions and human-annotated answers
- Supporting evidence
- Figures and tables

### Sample Record

| Field | Value |
|---|---|
| Paper ID | `1909.00694` |
| Title | Minimally Supervised Learning of Affective Events Using Discourse Relations |
| Sections | 21 |
| Questions | 9 |
| Extracted paragraphs | 56 |

**Sample question:** What is the seed lexicon?

**Gold answer:** A vocabulary of positive and negative predicates used to determine the polarity of an event.

**Gold evidence ID:** `1909.00694:4:1`

## 4. Implementation

### Document Ingestion

Implemented paragraph-level ingestion using LangChain `Document`.

Each document contains:

- Original paragraph text
- Paper ID and title
- Section name and index
- Paragraph index
- Stable document ID

Example ID: `1909.00694:4:1`

Validated document counts and preservation of gold evidence.

### Vector Indexing

- Embedding model: `text-embedding-3-small`
- Embedding dimensions: 1,536
- Vector database: Chroma with persistent local storage
- Indexed 10 research papers containing 975 paragraphs
- Implemented reusable indexing that skips existing document IDs

### Semantic Retrieval

Implemented top-K similarity search with `paper_id` metadata filtering.

Verified that retrieval returns documents only from the requested research paper.

### Grounded Generation

Implemented structured LLM generation using Pydantic.

The output includes:

```json
{
  "answer": "A seed lexicon consists of positive and negative predicates.",
  "citations": ["1909.00694:4:1"],
  "is_answerable": true
}
```

Also implemented citation ID validation and abstention when evidence is insufficient.

## 5. Evaluation

### Retrieval Evaluation — ir-measures

Evaluated six fully mapped questions from the first QASPER paper.

| Metric | Score |
|---|---:|
| Recall@5 | 0.500 |
| Recall@10 | 0.833 |
| MRR@5 | 0.250 |
| MRR@10 | 0.295 |

Three other questions were excluded because their annotated evidence could not be fully mapped to indexed paragraphs.

### Generation Evaluation — DeepEval

Evaluated five questions using an LLM judge.

| Metric | Average |
|---|---:|
| Faithfulness | 1.000 |
| Answer Relevancy | 0.767 |

Four of five answers met the configured relevancy threshold of 0.70.

## 6. Key Learnings

- Preserve document metadata for evidence citations.
- Use stable IDs for repeatable indexing.
- Filter retrieval by paper ID for paper-specific QA.
- Evaluate retrieval and generation separately.
- Higher retrieval depth improves recall but increases context size.
- Faithfulness does not guarantee answer relevancy or correctness.
- Missing table evidence creates ingestion coverage gaps.
- LLM evaluation scores can vary between runs.

## 7. Current Limitations

- Only 10 research papers indexed, not the full dataset.
- Paragraph-only ingestion; table contents are not indexed.
- Semantic retrieval without hybrid search or reranking.
- Evaluation performed on a small question sample.
- Citation validation verifies document IDs, not complete claim-level attribution.
