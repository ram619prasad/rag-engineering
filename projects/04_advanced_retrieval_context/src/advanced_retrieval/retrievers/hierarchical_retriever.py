
import json

from langchain_core.runnables import RunnableLambda

from advanced_retrieval.retrievers.document_retriever import (
    create_document_retriever,
)
from advanced_retrieval.retrievers.bm25_retriever import (
    CATALOG_PATH,
)
from advanced_retrieval.retrievers.semantic_retriever import (
    SemanticRetriever,
)


def create_hierarchical_retriever(
    sections: list[dict],
    document_k: int = 3,
    section_k: int = 10,
) -> RunnableLambda:
    # Level 0: Document retrieval
    document_retriever = create_document_retriever(
        sections=sections,
        k=document_k,
    )

    # Level 1: Reuse our existing section index
    section_retriever = SemanticRetriever(
        k=section_k,
    )

    if not section_retriever.has_index():
        raise RuntimeError(
            "Baseline semantic section index is missing. "
            "Build the Module 02 index first."
        )

    def retrieve(query: str):
        # Step 1: Find relevant documents
        documents = document_retriever.invoke(query)

        document_ids = list(dict.fromkeys(
            doc.metadata["document_id"]
            for doc in documents
        ))

        if not document_ids:
            return []

        # Step 2: Search sections ONLY in selected documents
        return section_retriever.vectorstore.similarity_search(
            query=query,
            k=section_k,
            filter={
                "document_id": {
                    "$in": document_ids,
                }
            },
        )

    return RunnableLambda(retrieve)


def main():
    sections = json.loads(
        CATALOG_PATH.read_text(encoding="utf-8")
    )

    retriever = create_hierarchical_retriever(
        sections=sections,
        document_k=3,
        section_k=10,
    )

    query = (
        "How does Kubernetes HPA manage replicas "
        "during rolling updates?"
    )

    results = retriever.invoke(query)

    print(f"\nQuery: {query}")
    print("\n=== HIERARCHICAL RETRIEVAL ===")

    for rank, section in enumerate(results, start=1):
        print(
            f"{rank}. {section.metadata['section_id']}"
        )
        print(
            f"   Document: {section.metadata['document_id']}"
        )
        print(
            f"   Text: {section.page_content[:150]}..."
        )


if __name__ == "__main__":
    main()
