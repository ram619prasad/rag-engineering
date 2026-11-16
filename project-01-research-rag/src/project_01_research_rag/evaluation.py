from langchain_core.documents import Document
from datasets import load_dataset
from project_01_research_rag.ingestion import paper_to_documents

dataset = load_dataset("allenai/qasper", split="train")
paper = dataset[0]
documents = paper_to_documents(paper)

# evidence_texts = paper["qas"]["answers"][0]["answer"][0]["evidence"]
        

def map_evidence_to_ids(
    evidence_texts: list[str],
    documents: list[Document],
) -> tuple[set[str], list[str]]:
    
    gold_ids: set[str] = set()
    unmatched_evidence: list[str] = []
    
    paragraph_to_id = {
        doc.page_content.strip(): doc.id
        for doc in documents
    }

    for text in evidence_texts:
        doc_id = paragraph_to_id.get(text.strip())
        if not doc_id:
            unmatched_evidence.append(text)
            continue

        gold_ids.add(doc_id)

    return (gold_ids, unmatched_evidence)


qas = paper["qas"]
for question_index, question in enumerate(qas["question"]):
    annotations = qas["answers"][question_index]["answer"]

    evidence_texts = []

    for annotation in annotations:
        evidence_texts.extend(annotation["evidence"])

    # IMPORTANT: This is outside the annotation loop.
    gold_ids, unmatched = map_evidence_to_ids(
        evidence_texts,
        documents,
    )

    print(f"Question {question_index}: {question}")
    print("Matched:", len(gold_ids))
    print("Unmatched:", len(unmatched))
