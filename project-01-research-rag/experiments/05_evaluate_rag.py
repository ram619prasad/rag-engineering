
from datasets import load_dataset
from ir_measures import Qrel, ScoredDoc, R, RR, calc_aggregate

from project_01_research_rag.ingestion import paper_to_documents
from project_01_research_rag.vector_store import create_vector_store
from project_01_research_rag.retrieval import retrieve_evidence
from project_01_research_rag.evaluation import map_evidence_to_ids


def main():
    K = 10
    
    # 1. Load the first QASPER paper
    dataset = load_dataset("allenai/qasper", split="train")
    paper = dataset[0]

    documents = paper_to_documents(paper)
    vector_store = create_vector_store()

    qas = paper["qas"]

    all_qrels = []
    all_run = []

    evaluated_questions = 0
    skipped_questions = 0

    # 2. Iterate through all questions
    for question_index, question in enumerate(qas["question"]):
        question_id = qas["question_id"][question_index]
        annotations = qas["answers"][question_index]["answer"]

        # 3. Collect evidence across all human annotations
        evidence_texts = []

        for annotation in annotations:
            evidence_texts.extend(annotation["evidence"])

        # 4. Map gold evidence to document IDs
        gold_ids, unmatched = map_evidence_to_ids(
            evidence_texts=evidence_texts,
            documents=documents,
        )

        # 5. Skip questions with incomplete evidence mapping
        if unmatched or not gold_ids:
            skipped_questions += 1
            print(f"SKIPPED Q{question_index}: {question}")
            continue

        evaluated_questions += 1

        # 6. Create relevance judgments
        for doc_id in gold_ids:
            all_qrels.append(
                Qrel(
                    query_id=question_id,
                    doc_id=doc_id,
                    relevance=1,
                )
            )

        # 7. Retrieve top-5 documents for this question
        retrieved_documents = retrieve_evidence(
            vector_store=vector_store,
            question=question,
            paper_id=paper["id"],
            k=K,
        )

        # 8. Convert ranked documents to ScoredDoc
        for rank, doc in enumerate(retrieved_documents, start=1):
            all_run.append(
                ScoredDoc(
                    query_id=question_id,
                    doc_id=str(doc.id),
                    score=K + 1 - rank,
                )
            )
            
        question_metrics = calc_aggregate(
            [R@5, R@10, RR@5, RR@10],
            [
                Qrel(question_id, doc_id, 1)
                for doc_id in gold_ids
            ],
            [
                ScoredDoc(
                    question_id,
                    str(doc.id),
                    K + 1 - rank,
                )
                for rank, doc in enumerate(retrieved_documents, start=1)
            ],
        )

        print(
            f"Q{question_index}: "
            f"R@5={question_metrics[R@5]:.2f}, "
            f"R@10={question_metrics[R@10]:.2f}, "
            f"RR@10={question_metrics[RR@10]:.2f}"
        )

        print(
            f"Q{question_index}: "
            f"Gold={len(gold_ids)}, "
            f"Retrieved={len(retrieved_documents)}"
        )
        
        

    # 9. Calculate aggregate retrieval metrics
    if evaluated_questions == 0:
        print("No questions eligible for evaluation.")
        return

    metrics = calc_aggregate(
        [R@5, R@10, RR@5, RR@10],
        all_qrels,
        all_run,
    )

    print("\n===== RETRIEVAL EVALUATION =====")
    print("Paper ID:", paper["id"])
    print("Evaluated questions:", evaluated_questions)
    print("Skipped questions:", skipped_questions)
    print("Recall@5:", metrics[R@5])
    print("MRR@5:", metrics[RR@5])
    print("Recall@10:", metrics[R@10])
    print("MRR@10:", metrics[RR@10])


if __name__ == "__main__":
    main()
