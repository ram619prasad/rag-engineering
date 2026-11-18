# Project 02 — Citation-Aware RAG

## Objective

Build a RAG system that generates answers from research papers and verifies whether the cited evidence actually supports each factual claim.

**Key improvement over Project 01:** Validating citation IDs is not enough. We must check whether the referenced passages support the claims.

## Tech Stack

- Python + uv
- QASPER dataset (Hugging Face)
- LangChain
- Chroma
- OpenAI embeddings (`text-embedding-3-small`)
- ChatOpenAI (`gpt-4.1-mini`)
- Pydantic
- Rich

## Architecture

```text
QASPER Dataset
      ↓
Document Ingestion
      ↓
OpenAI Embeddings + Chroma
      ↓
Semantic Retrieval
      ↓
Grounded Answer Generation
      ↓
Claim Extraction
      ↓
Original Citation Verification
      ↓
Supporting Evidence Discovery
      ↓
Citation Quality Evaluation
      ↓
Rich Execution Report
```

## Implementation

| Module | Implementation |
|---|---|
| `ingestion.py` | Convert research paragraphs into LangChain Documents |
| `vector_store.py` | Initialize persistent Chroma |
| `indexing.py` | Skip previously indexed document IDs |
| `retrieval.py` | Paper-filtered semantic similarity search |
| `generation.py` | Generate grounded answers, validate citations, and abstain when needed |
| `claim_extraction.py` | Extract independently verifiable factual statements |
| `verification.py` | Verify original citations and discover supporting passages |
| `evaluation.py` | Calculate citation precision and claim support coverage |
| `pipeline.py` | Orchestrate the reusable modules |
| `reporting.py` | Display execution results using Rich |
| `main.py` | Application entry point |

## Dataset and Experiments

**Dataset:** [allenai/qasper](https://huggingface.co/datasets/allenai/qasper)

Initial experiments used paper `1909.00694`.

- Extracted and indexed 56 paragraphs.
- Verified idempotent indexing.
- Retrieved five paragraphs for "What is the seed lexicon?"
- Found the annotated supporting evidence at rank 2.
- Generated grounded answers with valid citation IDs.
- Verified abstention for an unrelated question.
- Extracted individual claims from a generated answer.

## Citation Verification

Implemented three verification outcomes:

- **Supported:** Evidence establishes the claim.
- **Contradicted:** Evidence conflicts with the claim.
- **Insufficient evidence:** Evidence cannot establish or refute the claim.

All three outcomes were verified using controlled examples.

### Evidence Discovery Experiment

Claim: The seed lexicon contains 15 positive and 15 negative words.

- Original citation `1909.00694:4:1` — Insufficient evidence.
- Discovered citation `1909.00694:11:4` — Supported.

This demonstrated that a correct answer can have an incorrect citation.

## Evaluation

Implemented:

**Citation Precision**

Supported claim–citation pairs divided by total evaluated pairs.

**Claim Support Coverage**

Claims with at least one supporting citation divided by total claims.

Controlled experiment results:

| Metric | Score |
|---|---:|
| Citation precision | 0.50 |
| Claim support coverage | 0.667 |

A separate end-to-end QASPER run generated three claims and reported 1.0 for original citation precision, original claim coverage, and discovered claim coverage.

These are small-sample LLM-judged results, not general accuracy guarantees.

## Running the Project

Install dependencies:

```bash
uv sync
```

Configure your OpenAI API key using environment variables or a local `.env` file.

Run:

```bash
uv run main.py
```

The Rich report displays:

1. Dataset and indexing summary
2. Generated answer
3. Retrieved evidence
4. Extracted claims
5. Original citation verification
6. Supporting evidence discovery
7. Citation evaluation metrics

## Key Learnings

- A claim is an individual factual statement made by the LLM.
- A citation identifies the evidence used to support a claim.
- A valid citation ID does not guarantee actual support.
- Missing evidence is different from contradictory evidence.
- Citation auditing and evidence discovery solve different problems.
- Citation precision and claim support coverage measure different aspects of citation quality.
- Reusable modules can be orchestrated into one end-to-end application.

## Limitations

- Evaluated on a small number of QASPER examples.
- LLM-based verification can make mistakes.
- Evidence discovery searches only retrieved passages.
- Individual passages are verified separately, not as combined evidence.
- Verification requires multiple LLM calls and can increase latency and cost.
