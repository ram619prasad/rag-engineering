from project_01_research_rag.vector_store import create_vector_store
from project_01_research_rag.retrieval import retrieve_evidence
from project_01_research_rag.generation import generate_answer, validate_citations, GroundedAnswer

vector_store = create_vector_store()

question = "What is the seed lexicon?"
paper_id = "1909.00694"

documents = retrieve_evidence(
    vector_store=vector_store,
    question=question,
    paper_id=paper_id,
    k=5,
)

answer = generate_answer(
    question=question,
    documents=documents,
)

print(answer.model_dump_json(indent=2))

# Test 1: Real generated answer
assert validate_citations(answer, documents) is True

# Test 2: Fabricated citation
invalid_answer = answer.model_copy(
    update={"citations": ["1909.00694:99:99"]}
)
assert validate_citations(invalid_answer, documents) is False

# Test 3: Unanswerable response
unanswerable_answer = GroundedAnswer(
    answer="Insufficient evidence.",
    citations=[],
    is_answerable=False,
)
assert validate_citations(unanswerable_answer, documents) is True

# Empty Evidence
empty_answer = generate_answer(
    question="What is the seed lexicon?",
    documents=[],
)

print(empty_answer.model_dump_json(indent=2))

assert empty_answer.is_answerable is False
assert empty_answer.citations == []

# Irerelevant Evidence
unsupported_question = (
    "What was the exact annual revenue of the company in 2025?"
)

unsupported_answer = generate_answer(
    question=unsupported_question,
    documents=documents,
)

print(unsupported_answer.model_dump_json(indent=2))

assert unsupported_answer.is_answerable is False
assert unsupported_answer.citations == []
