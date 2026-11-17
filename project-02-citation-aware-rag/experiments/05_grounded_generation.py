from project_02_citation_aware_rag.vector_store import create_vector_store
from project_02_citation_aware_rag.retrieval import retrieve_evidence
from project_02_citation_aware_rag.generation import generate_answer

vector_store = create_vector_store()

paper_id = "1909.00694"

question = "What is the seed lexicon?"

documents = retrieve_evidence(
    vector_store=vector_store,
    question=question,
    paper_id=paper_id,
    k=5,
)

result = generate_answer(question, documents)

print(result.model_dump_json(indent=2))


unanswerable_question = "What was the company's revenue in 2025?"
result = generate_answer(
    question=unanswerable_question,
    documents=documents,
)
print(result.model_dump_json(indent=2))