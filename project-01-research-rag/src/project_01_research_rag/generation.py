from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

LLM = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)

class GroundedAnswer(BaseModel):
    answer: str = Field(
        description="Answer supported by retrieved evidence."
    )
    citations: list[str] = Field(
        description="IDs of the supporting evidence documents."
    )
    is_answerable: bool = Field(
        description="Whether the evidence supports an answer"
    )
    
def format_evidence(documents: list[Document]) -> str:
    evidence_blocks = []

    for document in documents:
        document_id = document.id
        section = document.metadata.get("section")
        paragraph = document.page_content
        
        formatted_block = (
            f"[{document_id}]\n"
            f"Section: {section}\n"
            f"Content: {paragraph}"
        )
        evidence_blocks.append(formatted_block)

    return "\n\n".join(evidence_blocks)

def validate_citations(
    answer: GroundedAnswer,
    documents: list[Document],
) -> bool:
    if answer.is_answerable and not answer.citations:
        return False

    if not answer.is_answerable and answer.citations:
        return False
    
    ids = ids = {doc.id for doc in documents}
    citation_ids = set(answer.citations)

    return citation_ids.issubset(ids)

def generate_answer(
    question: str,
    documents: list[Document],
) -> GroundedAnswer:
    if not documents:
        return GroundedAnswer(
            answer="Insufficient evidence in the retrieved passages.",
            citations=[],
            is_answerable=False,
        )
    
    # 1. Format the documents using format_evidence()
    context = format_evidence(documents)
    structured_llm = LLM.with_structured_output(
        GroundedAnswer,
        method="json_schema",
    )

    # 2. Create the prompt input using question and context
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a research assistant.

            Rules:
            - Answer using only the provided evidence.
            - Do not rely on external knowledge.
            - Cite the document IDs supporting your answer.
            - Never cite an ID not present in the evidence.
            - If the evidence is insufficient, set is_answerable
            to false, return an empty citations list, and
            explain that there is insufficient evidence.
            """
        ),
        (
            "human",
            "Question: {question}\n\nEvidence:\n{context}"
        ),
    ])

    # 3. Connect the prompt to structured_llm
    chain = prompt | structured_llm

    # 4. Invoke the chain
    result = chain.invoke({
        "question": question,
        "context": context,
    })

    # 5. Return the GroundedAnswer
    if not isinstance(result, GroundedAnswer):
        raise TypeError(
            f"Expected GroundedAnswer, got {type(result).__name__}"
        )
        
    if not result.is_answerable:
        return GroundedAnswer(
            answer="Insufficient evidence in the retrieved passages.",
            citations=[],
            is_answerable=False,
        )
    return result