## Implementation Progress

### 1. QASPER Ingestion
- Implemented reusable `paper_to_documents()` in `ingestion.py`.
- Converted research paper paragraphs into LangChain Documents.
- Preserved stable document IDs and section-level metadata.
- Verified extraction of 56 paragraphs from the first QASPER paper.

### 2. Vector Store & Indexing
- Configured persistent Chroma with OpenAI `text-embedding-3-small` embeddings.
- Created the `qasper_citation_aware` collection.
- Implemented reusable, idempotent indexing in `indexing.py`.
- Verified that repeated indexing skips all 56 existing documents.

### 3. Claim Extraction
- Defined Pydantic models for claims and verification results.
- Implemented structured claim extraction using LangChain and OpenAI.
- Successfully extracted three independently verifiable claims from a sample answer.
- Preserved candidate citation IDs for subsequent verification.

### 4. Semantic Retrieval

- Implemented reusable `retrieve_evidence()` in `retrieval.py`.
- Used Chroma similarity search with configurable Top-K and `paper_id` metadata filtering.
- Retrieved five relevant paragraphs for the question "What is the seed lexicon?"
- Found human-annotated evidence `1909.00694:4:1` at rank 2.
- Verified Hit@5 = 1 and Reciprocal Rank = 0.5 for the sample question.

### 5. Grounded Answer Generation

- Implemented reusable evidence formatting and grounded answer generation.
- Used LangChain, ChatOpenAI, and Pydantic structured output.
- Added citation ID validation and answerability consistency checks.
- Generated an evidence-grounded answer with two valid citations.
- Verified abstention when retrieved passages cannot answer the question.
- Normalized unanswerable responses to a readable message.

### 6. Integrated Claim Extraction

- Connected semantic retrieval, grounded generation, and structured claim extraction.
- Extracted independently verifiable claims from real QASPER-generated answers.
- Preserved candidate citation IDs for downstream verification.
- Observed that valid citation IDs can be incorrectly associated with factual claims.

**Learning:** Citation validity and citation correctness are different. Claim-level evidence verification is needed to determine whether retrieved passages genuinely support individual claims.

### 7. Citation Verification & Evidence Discovery

- Implemented `verify_claim()` to classify claims as supported, contradicted, or insufficiently supported.
- Implemented `verify_claims()` to audit claim–citation pairs against retrieved evidence.
- Implemented `find_supporting_citations()` to discover supporting passages.
- Verified all three verdicts using controlled examples.
- Detected incorrect citation attribution in real QASPER answers.
- Successfully identified `1909.00694:11:4` as the correct supporting citation for a claim initially associated with `1909.00694:4:1`.

**Learning:** Correct answers can contain incorrect citations. Citation auditing and evidence discovery are separate, complementary capabilities.