from project_02_citation_aware_rag.models import Claim
from project_02_citation_aware_rag.vector_store import create_vector_store
from project_02_citation_aware_rag.retrieval import retrieve_evidence
from project_02_citation_aware_rag.verification import find_supporting_citations


vector_store = create_vector_store()

documents = retrieve_evidence(
    vector_store=vector_store,
    question="What is the seed lexicon?",
    paper_id="1909.00694",
    k=5,
)

claim = Claim(
    text="The seed lexicon consists of 15 positive words and 15 negative words.",
    citation_ids=["1909.00694:4:1"],
)

results = find_supporting_citations(
    claim=claim,
    documents=documents,
)

print(f"Original citation: {claim.citation_ids}")
print(f"Supporting citations found: {len(results)}")

for result in results:
    print(f"\nCitation: {result.citation_ids}")
    print(f"Verdict: {result.verdict}")
    print(f"Reason: {result.reason}")