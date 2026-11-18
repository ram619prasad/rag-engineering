from project_02_citation_aware_rag.models import Claim, VerificationResult
from project_02_citation_aware_rag.evaluation import (
    citation_precision,
    claim_support_coverage,
)

claims = [
    Claim(text="Claim A", citation_ids=["doc1", "doc2"]),
    Claim(text="Claim B", citation_ids=["doc3"]),
    Claim(text="Claim C", citation_ids=["doc4"]),
]

results = [
    VerificationResult(
        claim="Claim A",
        citation_ids=["doc1"],
        verdict="supported",
        reason="Evidence supports Claim A.",
    ),
    VerificationResult(
        claim="Claim A",
        citation_ids=["doc2"],
        verdict="insufficient_evidence",
        reason="Evidence does not establish Claim A.",
    ),
    VerificationResult(
        claim="Claim B",
        citation_ids=["doc3"],
        verdict="supported",
        reason="Evidence supports Claim B.",
    ),
    VerificationResult(
        claim="Claim C",
        citation_ids=["doc4"],
        verdict="contradicted",
        reason="Evidence conflicts with Claim C.",
    ),
]

print("Citation precision:", citation_precision(results))
print("Claim support coverage:", claim_support_coverage(claims, results))