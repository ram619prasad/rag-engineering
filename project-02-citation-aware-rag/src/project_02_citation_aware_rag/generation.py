from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from project_02_citation_aware_rag.models import GroundedAnswer


def format_evidence(documents: list[Document]) -> str:
    evidence_blocks = []
    for doc in documents:
        evidence_blocks.append(
            f"[Document ID: {doc.id}]\n"
            f"Section: {doc.metadata["section"]}\n"
            f"Content: {doc.page_content}"
        )
        
    return "\n\n".join(evidence_blocks)

def generate_answer(
    question: str,
    documents: list[Document],
) -> GroundedAnswer:
    if not documents:
        return GroundedAnswer(
            answer="Insufficient evidence.",
            citation_ids=[],
            is_answerable=False,
        )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a research assistant answering questions
                using retrieved scientific evidence.

                Rules:
                1. Answer using ONLY the provided evidence.
                2. Do not invent facts.
                3. Cite supporting document IDs from the evidence.
                4. Never invent citation IDs.
                5. If evidence is insufficient, abstain.
                6. When abstaining, set is_answerable=False
                and return an empty citation_ids list.
                """,
            ),
            (
                "human",
                "Question:\n{question}\n\nEvidence:\n{evidence}",
            ),
        ]
    )
    
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )
    
    structured_llm = llm.with_structured_output(
        GroundedAnswer,
        method="json_schema",
    )
    
    chain = prompt | structured_llm
    
    evidence = format_evidence(documents)
    
    result = chain.invoke({
        "question": question,
        "evidence": evidence
    })
    
    return _validate_grounded_answer(result, documents)


def _validate_grounded_answer(
    result: object,
    documents: list[Document],
) -> GroundedAnswer:
    if not isinstance(result, GroundedAnswer):
        raise TypeError(
            f"Expected GroundedAnswer, got {type(result).__name__}"
        )

    # Validate the original output before normalizing it.
    if not result.is_answerable:
        if result.citation_ids:
            raise ValueError(
                "Unanswerable response must not contain citations"
            )

        return GroundedAnswer(
            answer="Insufficient evidence in the retrieved passages.",
            citation_ids=[],
            is_answerable=False,
        )

    if not result.answer.strip():
        raise ValueError(
            "Answerable response must contain a nonempty answer"
        )

    if not result.citation_ids:
        raise ValueError(
            "Answerable response must contain at least one citation"
        )

    allowed_ids = {doc.id for doc in documents}
    unknown_ids = set(result.citation_ids) - allowed_ids

    if unknown_ids:
        raise ValueError(
            f"LLM returned unknown citation IDs: {unknown_ids}"
        )

    return result