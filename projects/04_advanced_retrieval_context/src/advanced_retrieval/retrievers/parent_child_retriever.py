
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from advanced_retrieval.retrievers.bm25_retriever import (
    CATALOG_PATH,
)


load_dotenv()


ROOT = Path(__file__).resolve().parents[3]

CHROMA_DIR = (
    ROOT / "data" / "chroma" / "parent_child"
)


def create_parent_child_retriever(
    sections: list[dict],
    child_chunk_size: int = 400,
    child_chunk_overlap: int = 60,
    search_k: int = 40,
) -> ParentDocumentRetriever:

    if not 0 <= child_chunk_overlap < child_chunk_size:
        raise ValueError(
            "Child overlap must be smaller than chunk size"
        )

    # 1. Build parent documents from our existing sections.
    parents = [
        Document(
            id=section["section_id"],
            page_content=section["text"],
            metadata={
                "section_id": section["section_id"],
                "document_id": section["document_id"],
                "heading_path": " > ".join(
                    section["heading_path"]
                ),
            },
        )
        for section in sections
    ]

    # 2. Split parents into smaller children.
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
    )

    # 3. Persistent vector store for child embeddings.
    vectorstore = Chroma(
        collection_name=(
            f"engineering_parent_child_"
            f"{child_chunk_size}_{child_chunk_overlap}"
        ),
        embedding_function=OpenAIEmbeddings(
            model="text-embedding-3-small"
        ),
        persist_directory=str(CHROMA_DIR),
    )

    # 4. In-memory key-value store for full parent sections.
    docstore = InMemoryStore()

    # 5. Create the LangChain retriever.
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        id_key="parent_id",
        search_kwargs={"k": search_k},
    )

    # 6. Determine whether the child index already exists.
    expected_child_count = sum(
        len(child_splitter.split_documents([parent]))
        for parent in parents
    )

    indexed_child_count = len(
        vectorstore.get(include=[])["ids"]
    )

    if indexed_child_count == 0:
        print(
            f"Indexing {expected_child_count} child chunks "
            f"from {len(parents)} parents..."
        )

        retriever.add_documents(
            documents=parents,
            ids=[parent.metadata["section_id"] for parent in parents],
        )

    elif indexed_child_count == expected_child_count:
        print(
            f"Reusing {indexed_child_count} indexed child chunks..."
        )

        # Chroma persists across runs, but InMemoryStore does not.
        # Restore all original parent documents by their stable IDs.
        docstore.mset([
            (parent.metadata["section_id"], parent)
            for parent in parents
        ])

    else:
        raise RuntimeError(
            "Existing child index appears incomplete or incompatible: "
            f"found {indexed_child_count}, "
            f"expected {expected_child_count}. "
            "Inspect or reset the parent-child collection "
            "before rebuilding."
        )

    return retriever


def main():
    sections = json.loads(
        CATALOG_PATH.read_text(encoding="utf-8")
    )

    retriever = create_parent_child_retriever(
        sections=sections,
        child_chunk_size=400,
        child_chunk_overlap=60,
        search_k=40,
    )

    query = (
        "How does Kubernetes HPA manage replicas "
        "during rolling updates?"
    )

    results = retriever.invoke(query)

    print(f"\nQuery: {query}")
    print(f"Unique parents returned: {len(results)}")

    print("\n=== TOP 5 PARENTS ===")

    for rank, parent in enumerate(results[:5], start=1):
        print(
            f"\n{rank}. {parent.metadata['section_id']}"
        )
        print(
            f"   Length: {len(parent.page_content)} characters"
        )
        print(
            f"   Preview: {parent.page_content[:180]}..."
        )


if __name__ == "__main__":
    main()
