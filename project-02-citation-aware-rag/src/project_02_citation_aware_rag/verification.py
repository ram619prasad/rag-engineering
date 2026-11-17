from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from project_02_citation_aware_rag.models import (
    Claim,
    VerificationResult,
)

load_dotenv()


def verify_claim(
    claim: Claim,
    evidence: str,
) -> VerificationResult:
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an expert at verifying factual claims against
            supporting evidence from scientific research papers.

            Your task is to determine whether the provided evidence
            supports, contradicts, or is insufficient to verify a claim.

            Verification Rules:

            1. Use ONLY the provided evidence.
            Do not rely on external knowledge or assumptions.

            2. Return exactly one of these verdicts:

            - supported:
                The evidence directly or logically supports the claim.

            - contradicted:
                The evidence explicitly conflicts with the claim.

            - insufficient_evidence:
                The evidence neither supports nor contradicts the claim,
                or does not contain enough information to verify it.

            3. Do not confuse missing evidence with contradiction.

            4. Evaluate the entire claim.
            If the evidence supports only part of a claim,
            return insufficient_evidence unless another part
            is explicitly contradicted.

            5. Pay attention to numbers, quantities, dates,
            comparisons, and factual relationships.

            6. Provide a short, clear reason explaining your verdict.
            Refer to the specific evidence that influenced your decision.

            7. Do not modify the original claim or invent citation IDs.

            Return the result using the required structured output schema.
            """,
        ),
        (
            "human",
            "Claim:\n{claim}\n\n"
            "Evidence:\n{evidence}",
        )
    ])
    
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )

    structured_llm = llm.with_structured_output(
        VerificationResult,
        method="json_schema",
    )

    chain = prompt | structured_llm
    
    result = chain.invoke(
        {
            "claim": claim.text,
            "evidence": evidence,
        }
    )
    
    if not isinstance(result, VerificationResult):
        raise TypeError(
            f"Expected result to be VerificationResult but got {type(result).__name__}"
        )
    
    return VerificationResult(
        claim=claim.text,
        citation_ids=claim.citation_ids,
        verdict=result.verdict,
        reason=result.reason,
    )
    
    
def verify_claims(
    claims: list[Claim],
    documents: list[Document],
) -> list[VerificationResult]:
    # Step 1: Build a lookup of retrieved documents
    document_lookup = {
        doc.id: doc
        for doc in documents
    }

    results: list[VerificationResult] = []

    # Step 2: Process each extracted claim
    for claim in claims:

        # Step 3: Verify each candidate citation independently
        for citation_id in dict.fromkeys(claim.citation_ids):

            # Find the corresponding evidence
            document = document_lookup.get(citation_id)

            # Step 4: Handle missing evidence
            if document is None:
                results.append(
                    VerificationResult(
                        claim=claim.text,
                        citation_ids=[citation_id],
                        verdict="insufficient_evidence",
                        reason="Citation not found in retrieved documents.",
                    )
                )
                continue

            # Step 5: Create a claim for this citation only
            single_citation_claim = Claim(
                text=claim.text,
                citation_ids=[citation_id],
            )

            # Step 6: Reuse our existing verifier
            result = verify_claim(
                claim=single_citation_claim,
                evidence=document.page_content,
            )

            results.append(result)

        # Handle claims without any candidate citations
        if not claim.citation_ids:
            results.append(
                VerificationResult(
                    claim=claim.text,
                    citation_ids=[],
                    verdict="insufficient_evidence",
                    reason="No candidate citations were provided.",
                )
            )

    return results
    
    
def find_supporting_citations(
    claim: Claim,
    documents: list[Document],
) -> list[VerificationResult]:
    supporting_results: list[VerificationResult] = []
    
    for document in documents:
        if document.id is None:
            raise ValueError("Retrieved document has no ID")
    
        single_citation_claim = Claim(
            text=claim.text,
            citation_ids=[document.id],
        )
        
        verified_claim: VerificationResult = verify_claim(single_citation_claim, document.page_content)
        
        if verified_claim.verdict == "supported":
            supporting_results.append(verified_claim)
    
    return supporting_results