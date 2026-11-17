from langchain_chroma import Chroma
from langchain_core.documents import Document


def retrieve_evidence(
    vector_store: Chroma,
    question: str,
    paper_id: str,
    k: int = 5,
) -> list[Document]:
    return vector_store.similarity_search(
        query=question,
        k=k,
        filter={
            "paper_id": paper_id
        }
    )