from datasets import load_dataset

from project_02_citation_aware_rag.ingestion import paper_to_documents
from project_02_citation_aware_rag.vector_store import create_vector_store
from project_02_citation_aware_rag.indexing import index_documents
from project_02_citation_aware_rag.retrieval import retrieve_evidence
from project_02_citation_aware_rag.generation import generate_answer
from project_02_citation_aware_rag.claim_extraction import extract_claims
from project_02_citation_aware_rag.models import PipelineResult, ExtractedClaims, VerificationResult
from project_02_citation_aware_rag.verification import (
    verify_claims,
    find_supporting_citations,
)
from project_02_citation_aware_rag.evaluation import (
    citation_precision,
    claim_support_coverage,
)


def run_pipeline(
    question: str,
    paper_index: int = 0,
    k: int = 5,
) -> PipelineResult:
    # Load Data set & specific paper
    dataset = load_dataset("allenai/qasper", split="train")
    paper = dataset[paper_index]
    
    # Document conversion
    documents = paper_to_documents(paper=paper)
    
    # Indexing
    vector_store = create_vector_store()
    newly_indexed = index_documents(
        vector_store=vector_store,
        documents=documents,
    )
    
    # Retrieve
    retrieve_documents = retrieve_evidence(
        vector_store=vector_store,
        question=question,
        paper_id=paper["id"],
        k=k
    )
    
    # Grounded Answer generation
    grounded_answer = generate_answer(
        question=question,
        documents=retrieve_documents
    )
    
    # Extract claims
    if grounded_answer.is_answerable:
        extracted_claims = extract_claims(
            answer=grounded_answer.answer,
            citation_ids=grounded_answer.citation_ids,
        ).claims
    else:
        extracted_claims = []
        
    if not extract_claims:
        raise RuntimeError("Error processing....")
    
    
    # extracted_claim = extract_claims(
    #     answer=grounded_answer.answer,
    #     citation_ids=grounded_answer.citation_ids
    # )
    
    original_ver_results = verify_claims(extracted_claims, retrieve_documents)
    cit_precision = citation_precision(original_ver_results)
    claim_coverage = claim_support_coverage(extracted_claims, original_ver_results)
    
    # Discover supporting citations for every extracted claim
    discovered_results: list[VerificationResult] = []

    for claim in extracted_claims:
        supporting_results = find_supporting_citations(
            claim=claim,
            documents=retrieve_documents,
        )

        discovered_results.extend(supporting_results)
    
    discovered_coverage = claim_support_coverage(
        claims=extracted_claims,
        results=discovered_results,
    )
    
    return PipelineResult(
        paper_id=paper["id"],
        paper_title=paper["title"],
        documents_indexed=vector_store._collection.count(),
        newly_indexed=newly_indexed,
        question=question,
        retrieved_documents=retrieve_documents,
        grounded_answer=grounded_answer,
        claims=extracted_claims,
        original_verifications=original_ver_results,
        discovered_verifications=original_ver_results,
        original_citation_precision=cit_precision,
        original_claim_coverage=claim_coverage,
        discovered_claim_coverage=discovered_coverage,
    )
