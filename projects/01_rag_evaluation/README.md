# Project 01 — RAG Evaluation Foundation

## Objective

Build a practical evaluation pipeline to measure the quality of a RAG system's retrieval and generated answers.

The goal is to identify where a RAG system fails and compare improvements using measurable results instead of guesswork.

## Learning Tasks

### Module 1: Evaluation Fundamentals
- [ ] Understand why RAG evaluation is necessary
- [ ] Understand retrieval failures vs. generation failures
- [ ] Understand the difference between offline and online evaluation

### Module 2: Evaluation Dataset
- [ ] Define an evaluation dataset structure
- [ ] Create representative questions and expected answers
- [ ] Identify the relevant source chunks for each question

### Module 3: Retrieval Metrics
- [ ] Understand Precision@K
- [ ] Understand Recall@K
- [ ] Understand Hit Rate@K
- [ ] Implement the metrics using Python
- [ ] Manually verify the calculations with sample data

### Module 4: Answer Quality Metrics
- [ ] Understand answer correctness
- [ ] Understand faithfulness to retrieved context
- [ ] Understand answer relevance
- [ ] Identify how retrieval quality affects answer quality

### Module 5: Evaluation Pipeline
- [ ] Run retrieval against the evaluation dataset
- [ ] Calculate retrieval metrics
- [ ] Evaluate generated answers
- [ ] Produce a readable evaluation report

### Module 6: Compare and Improve
- [ ] Establish a baseline
- [ ] Compare two retrieval configurations
- [ ] Identify failure cases and explain the results
- [ ] Document findings and limitations

## Key Decisions

- Use the existing repository and `uv` environment.
- Learn concepts before implementing them.
- Prefer small, practical exercises over unnecessary abstractions.
- Manually verify results; automated tests are out of scope for now.
- Commit and push each meaningful topic separately.

## Completion Criteria

The project is complete when the evaluation pipeline can measure retrieval quality, assess generated answers, compare configurations, and produce a useful report.