### 01 — Dataset Setup and Inspection

**Implemented**
- Configured local QASPER storage using Hugging Face `save_to_disk()` and `load_from_disk()`.
- Implemented reusable `load_qasper()` in `dataset.py`.
- Inspected the first research paper (`1909.00694`).

**Verified**
- Dataset saved locally under `data/qasper/`.
- Subsequent executions load the dataset from local storage.
- First paper contains 9 questions: 3 with multiple human annotations and 6 with one annotation.

**Learning:** QASPER questions may have multiple valid human annotations, which must be considered during retrieval and answer evaluation.

### 02 — Evaluation Question Preparation

**Implemented**
- Created the `EvaluationQuestion` Pydantic model.
- Implemented `prepare_evaluation_question()` to extract questions and human annotations.
- Extracted reference answers from `free_form_answer` and `extractive_spans`.
- Collected and deduplicated gold evidence paragraphs.
- Included answerability information.

**Verified**

For Question 0 — *What is the seed lexicon?*

- Reference answers: `2`
- Unique gold evidence paragraphs: `1`
- Unanswerable: `False`

**Learning:** Gold answers describe what the correct answer should be, while gold evidence identifies the original supporting passages.

### 03 — Paragraph Ingestion and Gold Evidence Mapping

**Implemented**
- Implemented `paper_to_documents()` to convert QASPER paragraphs into LangChain documents.
- Generated stable document IDs using `paper_id:section_index:paragraph_index`.
- Preserved paper, section, and paragraph metadata.
- Implemented `map_gold_evidence()` to match human-annotated paragraphs to document IDs.
- Returned unmatched evidence separately for evaluation diagnostics.

**Verified**

For Question 0 — *What is the seed lexicon?*

- Matched gold document ID: `1909.00694:4:1`
- Unmatched evidence paragraphs: `0`
- Result: Passed

**Learning:** Mapping gold evidence to stable document IDs enables comparisons with retrieved documents when calculating retrieval metrics such as Recall@K and MRR.

### Next — Persistent Vector Store and Retrieval

- Configure Chroma with OpenAI embeddings.
- Implement idempotent document indexing.
- Retrieve top-K passages for a question.
- Compare retrieved document IDs against gold evidence.


### 04.1 — Persistent Chroma Initialization

**Implemented**
- Created reusable `create_vector_store()` in `vector_store.py`.
- Configured OpenAI `text-embedding-3-small` embeddings.
- Initialized a persistent Chroma vector database under `data/chroma/`.
- Configured the `qasper` collection.

**Verified**
- Collection: `qasper`
- Indexed documents: `0`
- Result: Passed

**Learning:** Vector store initialization and document indexing are separate responsibilities. A persistent Chroma collection can be reopened across executions without rebuilding its contents.

**Next:** Implement idempotent document indexing.

### 04.2 — Idempotent Document Indexing

**Implemented**
- Created reusable `index_documents()` in `indexing.py`.
- Identified existing document IDs using Chroma's `get()` API.
- Indexed only documents not already present in the collection.
- Added validation for missing document IDs.
- Skipped insertion when no new documents were found.
- Returned the number of newly indexed documents.

**Verified**
- Documents prepared: `56`
- First indexing run: `56` newly indexed
- Second indexing call: `0` newly indexed
- Subsequent program execution: `0` newly indexed
- Final collection count: `56`
- Result: Passed

**Learning:** Idempotent indexing prevents unnecessary embedding API calls, while persistent storage allows experiments to reuse the same document collection.

**Next:** Implement semantic retrieval with paper-level filtering.

### 04.3 — Semantic Retrieval ✅

**Implemented**
- Created reusable `retrieve_evidence()` in `retrieval.py`.
- Used Chroma's `similarity_search()` to retrieve top-K paragraphs.
- Applied `paper_id` metadata filtering to restrict retrieval to the target research paper.

**Verified**
- Question: `What is the seed lexicon?`
- Documents retrieved: `5`
- Gold evidence document: `1909.00694:4:1`
- Gold evidence rank: `2`
- Result: Passed

**Learning:** Successful retrieval means the required evidence appears among the retrieved results. Its position also matters: ranking metrics help distinguish evidence retrieved first from evidence buried lower in the results.

**Next:** Implement reusable Recall@K and Reciprocal Rank evaluation.