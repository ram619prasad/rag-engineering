from pydantic import BaseModel
from typing import Literal

class GroundedAnswer(BaseModel):
    answer: str
    citation_ids : list[str]
    is_answerable: bool
    

class Claim(BaseModel):
    text: str
    citation_ids : list[str]
    

class VerificationResult(BaseModel):
    claim: str
    citation_ids: list[str]
    verdict: Literal["supported", "contradicted", "insufficient_evidence"]
    reason: str
    

class ExtractedClaims(BaseModel):
    claims: list[Claim]
    
from langchain_core.documents import Document


class PipelineResult(BaseModel):
    paper_id: str
    paper_title: str
    documents_indexed: int
    newly_indexed: int
    question: str
    retrieved_documents: list[Document]
    grounded_answer: GroundedAnswer
    claims: list[Claim]
    original_verifications: list[VerificationResult]
    discovered_verifications: list[VerificationResult]
    original_citation_precision: float
    original_claim_coverage: float
    discovered_claim_coverage: float