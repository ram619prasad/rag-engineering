from project_03_rag_evaluation_lab.dataset import load_qasper
from project_03_rag_evaluation_lab.vector_store import create_vector_store
from project_03_rag_evaluation_lab.retrieval import retrieve_evidence

paper = load_qasper()[0]
vector_store = create_vector_store()

documents = retrieve_evidence(
    vector_store=vector_store,
    question="What is the seed lexicon?",
    paper_id=paper["id"],
    k=5,
)

for rank, document in enumerate(documents, start=1):
    print(f"Rank {rank}: {document.id}")