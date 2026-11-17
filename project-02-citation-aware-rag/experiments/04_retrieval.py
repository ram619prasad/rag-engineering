from langchain_core.documents import Document

from project_02_citation_aware_rag.vector_store import create_vector_store
from project_02_citation_aware_rag.retrieval import retrieve_evidence

vector_store = create_vector_store()

question = "What is the seed lexicon?"
paper_id = "1909.00694"

documents = retrieve_evidence(
    vector_store=vector_store,
    question=question,
    paper_id=paper_id,
    k=5
)

print(f"Question: {question}")
print(f"Retrieved documents: {len(documents)}")

for rank, doc in enumerate(documents, start=1):
    print(f"\nRank {rank}")
    print(f"Document ID: {doc.id}")
    print(f"Section: {doc.metadata['section']}")
    print(f"Content: {doc.page_content[:250]}")