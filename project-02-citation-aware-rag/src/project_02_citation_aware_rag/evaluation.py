from project_02_citation_aware_rag.models import VerificationResult, Claim


def citation_precision(
    results: list[VerificationResult],
) -> float:
    if not results:
        return 0.0
    
    supported_count = sum(
        result.verdict == "supported"
        for result in results
    )
            
    return supported_count/len(results)


def claim_support_coverage(
    claims: list[Claim],
    results: list[VerificationResult],
) -> float:
    if not claims:
        return 0.0
    
    supported_claims = {
        result.claim
        for result in results
        if result.verdict == "supported"
    }
    
    return len(supported_claims)/len(claims)