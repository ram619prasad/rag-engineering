from project_02_citation_aware_rag.models import ExtractedClaims
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def extract_claims(
    answer: str,
    citation_ids: list[str],
) -> ExtractedClaims:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at extracting independently verifiable "
                "factual claims from generated answers. "
                "Extraction Rules:"
                "1. Extracts factual claims, not opinions or filler."        
                "2. Makes each claim independently understandable."
                "3. Preserves the meaning of the original answer."
                "4. Uses only candidate citation IDs provided in the input."
                "5. Does not claim that citations have been verified."
            ),
            (
                "human",
                "Answer:\n{answer}\n\n"
                "Candidate citation IDs:\n{citation_ids}",
            ),
        ]
    )
    
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )
    
    structured_llm = llm.with_structured_output(
        ExtractedClaims,
        method="json_schema",
    )
    
    chain = prompt | structured_llm
    
    result = chain.invoke(
        {
            "answer": answer, 
            "citation_ids": citation_ids,
        }
    )
    
    if not isinstance(result, ExtractedClaims):
        raise TypeError("Expected ExtractedClaims from structured LLM output")
    
    return result


answer = (
    "The seed lexicon contains 30 words. "
    "It includes 15 positive and 15 negative words."
)

citation_ids = ["1909.00694:11:4"]

result = extract_claims(
    answer=answer,
    citation_ids=citation_ids,
)

for index, claim in enumerate(result.claims, start=1):
    print(f"Claim {index}: {claim.text}")
    print(f"Candidate citations: {claim.citation_ids}")