from langchain_chroma import Chroma
from langchain_core.documents import Document


def index_documents(
    vector_store: Chroma,
    documents: list[Document],
) -> int:
    # Identify which document IDs already exist.
    document_ids = [str(doc.id) for doc in documents]
    existing = vector_store.get(
        ids=document_ids,
    )
    existing_ids = set(existing["ids"])

    # Keep only new documents.
    documents_to_index = [
        document
        for document in documents
        if document.id not in existing_ids
    ]
    if not documents_to_index:
        return 0

    # Add new documents to Chroma.
    vector_store.add_documents(
        documents=documents_to_index,
        ids=[document.id for document in documents_to_index]
    )

    # Return the number of documents added.
    return len(documents_to_index)