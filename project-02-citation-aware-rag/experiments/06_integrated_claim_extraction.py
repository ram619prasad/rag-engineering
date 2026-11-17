from project_02_citation_aware_rag.vector_store import create_vector_store
from project_02_citation_aware_rag.retrieval import retrieve_evidence
from project_02_citation_aware_rag.generation import generate_answer
from project_02_citation_aware_rag.claim_extraction import extract_claims

vector_store = create_vector_store()

paper_id = "1909.00694"

question = "What is the seed lexicon?"

documents = retrieve_evidence(
    vector_store=vector_store,
    question=question,
    paper_id=paper_id,
    k=5,
)

grounded_answer = generate_answer(question, documents)

if grounded_answer.is_answerable:
    extracted = extract_claims(
        answer=grounded_answer.answer,
        citation_ids=grounded_answer.citation_ids,
    )

    for index, claim in enumerate(extracted.claims, start=1):
        print(f"Claim {index}: {claim.text}")
        print(f"Candidate citations: {claim.citation_ids}")

