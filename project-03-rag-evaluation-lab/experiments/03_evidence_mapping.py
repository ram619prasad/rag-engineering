from project_03_rag_evaluation_lab.dataset import load_qasper
from project_03_rag_evaluation_lab.ingestion import paper_to_documents
from project_03_rag_evaluation_lab.evaluation_dataset import (
    prepare_evaluation_question,
)
from project_03_rag_evaluation_lab.evidence_mapping import map_gold_evidence

paper = load_qasper()[0]

documents = paper_to_documents(paper)
question = prepare_evaluation_question(paper, question_index=0)

matched_ids, unmatched = map_gold_evidence(
    gold_evidence=question.gold_evidence,
    documents=documents,
)

print("Question:", question.question)
print("Matched IDs:", matched_ids)
print("Unmatched evidence:", len(unmatched))