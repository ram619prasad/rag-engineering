from langchain_chroma import Chroma
from langchain_core.documents import Document


def index_documents(
    vector_store: Chroma,
    documents: list[Document],
) -> int:
    if not documents:
        return 0
    
    document_ids = [doc.id for doc in documents if doc.id is not None]
    existing_ids = set(vector_store.get(ids=document_ids)["ids"])
    
    documents_to_index = [
        doc
        for doc in documents 
        if doc.id not in existing_ids
    ]
    
    if documents_to_index:
        vector_store.add_documents(
            documents=documents_to_index,
            ids=[doc.id for doc in documents_to_index],
        )

    
    return len(documents_to_index)