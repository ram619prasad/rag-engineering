
import json
from pathlib import Path

from advanced_retrieval.retrievers.bm25_retriever import (
    BM25Retriever,
    CATALOG_PATH,
)
from advanced_retrieval.retrievers.semantic_retriever import (
    SemanticRetriever,
)
from advanced_retrieval.retrievers.hybrid_retriever import (
    create_hybrid_retriever,
)


DATA_DIR = CATALOG_PATH.parent
CONTEXTUAL_PATH = DATA_DIR / "contextual_sections.json"

CHROMA_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "chroma"
    / "contextual_baseline"
)


def load_contextual_sections() -> list[dict]:
    original = json.loads(
        CATALOG_PATH.read_text(encoding="utf-8")
    )

    contextual = json.loads(
        CONTEXTUAL_PATH.read_text(encoding="utf-8")
    )["sections"]

    original_ids = {
        section["section_id"] for section in original
    }

    contextual_ids = {
        section["section_id"] for section in contextual
    }

    if (
        len(contextual) != len(original)
        or len(contextual_ids) != len(contextual)
        or original_ids != contextual_ids
    ):
        raise ValueError(
            "Contextual corpus is incomplete or contains "
            "duplicate/mismatched section IDs."
        )

    # Our existing retriever classes read section["text"].
    # Supply contextual text for indexing without changing
    # the original section catalog.
    return [
        {
            **section,
            "text": section["contextual_text"],
        }
        for section in contextual
    ]


def main():
    sections = load_contextual_sections()

    # Contextual BM25
    bm25 = BM25Retriever(
        sections=sections,
        k=20,
    )

    # Separate contextual semantic index
    semantic = SemanticRetriever(
        k=20,
        collection_name="engineering_contextual_baseline",
        persist_directory=CHROMA_DIR,
    )

    # Check exact stored IDs, not just whether the index
    # contains at least one document.
    expected_ids = {s["section_id"] for s in sections}

    existing_docs = semantic.vectorstore.get_by_ids(
        list(expected_ids)
    )

    existing_ids = {doc.id for doc in existing_docs}

    missing_sections = [
        section for section in sections
        if section["section_id"] not in existing_ids
    ]

    if missing_sections:
        print(
            f"Indexing {len(missing_sections)} "
            "missing contextual sections..."
        )
        semantic.index(missing_sections)
    else:
        print("Reusing contextual vector index...")

    # Reuse our Module 02 Hybrid implementation
    hybrid = create_hybrid_retriever(
        bm25=bm25,
        semantic=semantic,
        bm25_weight=0.5,
        semantic_weight=0.5,
        rrf_k=60,
    )

    query = (
        "How does Kubernetes automatically increase "
        "replicas when CPU usage is high?"
    )

    for name, retriever in [
        ("CONTEXTUAL BM25", bm25),
        ("CONTEXTUAL SEMANTIC", semantic),
        ("CONTEXTUAL HYBRID", hybrid),
    ]:
        print(f"\n=== {name} ===")

        results = retriever.invoke(query)[:5]

        for rank, doc in enumerate(results, start=1):
            print(
                f"{rank}. {doc.metadata['section_id']}"
            )


if __name__ == "__main__":
    main()
