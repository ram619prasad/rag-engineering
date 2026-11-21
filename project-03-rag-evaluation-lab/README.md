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