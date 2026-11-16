from project_01_research_rag.vector_store import create_vector_store
from project_01_research_rag.retrieval import retrieve_evidence

vector_store = create_vector_store()

paper_id = "2003.07723"

results = retrieve_evidence(
    vector_store=vector_store,
    question="What is the seed lexicon?",
    paper_id=paper_id,
    k=5,
)

assert all(
    doc.metadata["paper_id"] == paper_id
    for doc in results
)

print("Retrieved documents:", len(results))
print("Paper filter validation passed!")