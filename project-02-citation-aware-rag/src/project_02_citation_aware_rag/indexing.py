from langchain_chroma import Chroma
from langchain_core.documents import Document


def index_documents(
    vector_store: Chroma,
    documents: list[Document],
) -> int:
    document_ids = [str(document.id) for document in documents]
    existing = vector_store.get(ids=document_ids)
    existing_ids = set(existing["ids"])
    
    docs_to_index = [
        doc 
        for doc in documents
        if doc.id not in existing_ids
    ]
    
    if docs_to_index:
        vector_store.add_documents(
            documents=docs_to_index,
            ids=[doc.id for doc in docs_to_index],
        )
    
    return len(docs_to_index)