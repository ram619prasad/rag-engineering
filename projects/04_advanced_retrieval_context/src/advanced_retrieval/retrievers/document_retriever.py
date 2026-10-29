
import json
from collections import defaultdict
from pathlib import Path

from advanced_retrieval.retrievers.bm25_retriever import (
    CATALOG_PATH,
)
from advanced_retrieval.retrievers.semantic_retriever import (
    SemanticRetriever,
)


ROOT = Path(__file__).resolve().parents[3]

CHROMA_DIR = (
    ROOT / "data" / "chroma" / "hierarchical"
)

COLLECTION_NAME = "engineering_document_level"


def build_document_representations(
    sections: list[dict],
) -> list[dict]:
    grouped = defaultdict(list)

    for section in sections:
        grouped[section["document_id"]].append(section)

    documents = []

    for document_id, document_sections in sorted(grouped.items()):
        parts = [
            f"Document: {document_id.replace('_', ' ')}"
        ]

        for section in document_sections:
            heading = " > ".join(section["heading_path"])

            if heading:
                parts.append(f"Section: {heading}")

            # Small excerpt gives the document representation
            # some content beyond its section headings.
            excerpt = " ".join(
                section["text"].split()
            )[:200]

            parts.append(excerpt)

        representation = "\n".join(parts)[:12000]

        documents.append({
            "section_id": document_id,
            "document_id": document_id,
            "heading_path": [],
            "text": representation,
        })

    return documents


def create_document_retriever(
    sections: list[dict],
    k: int = 3,
) -> SemanticRetriever:
    documents = build_document_representations(sections)

    retriever = SemanticRetriever(
        k=k,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    expected_ids = {
        doc["document_id"]
        for doc in documents
    }

    existing = retriever.vectorstore.get(
        ids=list(expected_ids),
        include=[],
    )

    existing_ids = set(existing["ids"])

    missing = [
        doc for doc in documents
        if doc["document_id"] not in existing_ids
    ]

    if missing:
        print(
            f"Indexing {len(missing)} document representations..."
        )
        retriever.index(missing)
    else:
        print("Reusing document-level index...")

    return retriever


def main():
    sections = json.loads(
        CATALOG_PATH.read_text(encoding="utf-8")
    )

    retriever = create_document_retriever(
        sections=sections,
        k=3,
    )

    query = (
        "How does Kubernetes HPA manage replicas "
        "during rolling updates?"
    )

    results = retriever.invoke(query)

    print(f"\nQuery: {query}")
    print("\n=== DOCUMENT-LEVEL RETRIEVAL ===")

    for rank, document in enumerate(results, start=1):
        print(
            f"{rank}. {document.metadata['document_id']}"
        )
        print(
            f"   Representation length: "
            f"{len(document.page_content)} characters"
        )


if __name__ == "__main__":
    main()
