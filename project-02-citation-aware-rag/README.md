## Implementation Progress

### 1. QASPER Ingestion — Completed
- Implemented reusable `paper_to_documents()` in `ingestion.py`.
- Converted research paper paragraphs into LangChain Documents.
- Preserved stable document IDs and section-level metadata.
- Verified extraction of 56 paragraphs from the first QASPER paper.

### 2. Vector Store & Indexing — Completed
- Configured persistent Chroma with OpenAI `text-embedding-3-small` embeddings.
- Created the `qasper_citation_aware` collection.
- Implemented reusable, idempotent indexing in `indexing.py`.
- Verified that repeated indexing skips all 56 existing documents.

### 3. Claim Extraction — Completed
- Defined Pydantic models for claims and verification results.
- Implemented structured claim extraction using LangChain and OpenAI.
- Successfully extracted three independently verifiable claims from a sample answer.
- Preserved candidate citation IDs for subsequent verification.

### 4. Semantic Retrieval
