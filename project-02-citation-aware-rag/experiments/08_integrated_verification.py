from project_02_citation_aware_rag.vector_store import create_vector_store
from project_02_citation_aware_rag.retrieval import retrieve_evidence
from project_02_citation_aware_rag.generation import generate_answer
from project_02_citation_aware_rag.claim_extraction import extract_claims
from project_02_citation_aware_rag.verification import verify_claims


vector_store = create_vector_store()

question = "What is the seed lexicon?"
paper_id = "1909.00694"

documents = retrieve_evidence(
    vector_store=vector_store,
    question=question,
    paper_id=paper_id,
    k=5,
)

# grounded_answer = generate_answer(question, documents)

# if grounded_answer.is_answerable:
#     extracted = extract_claims(
#         answer=grounded_answer.answer,
#         citation_ids=grounded_answer.citation_ids,
#     )

#     results = verify_claims(
#         claims=extracted.claims,
#         documents=documents,
#     )

#     for result in results:
#         print(result.model_dump_json(indent=2))

from project_02_citation_aware_rag.models import Claim
from project_02_citation_aware_rag.verification import verify_claim

claim = Claim(
    text="The seed lexicon consists of 15 positive words and 15 negative words.",
    citation_ids=["1909.00694:11:4"],
)

document_lookup = {doc.id: doc for doc in documents}

document = document_lookup["1909.00694:11:4"]

result = verify_claim(
    claim=claim,
    evidence=document.page_content,
)

print(result.model_dump_json(indent=2))