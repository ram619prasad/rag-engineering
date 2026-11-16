from pydantic import BaseModel
from typing import Literal

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