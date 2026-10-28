
import json

from langchain_classic.retrievers import EnsembleRetriever

from advanced_retrieval.retrievers.bm25_retriever import (
    BM25Retriever,
    CATALOG_PATH,
)
from advanced_retrieval.retrievers.semantic_retriever import (
    SemanticRetriever,
)


def create_hybrid_retriever(
    bm25: BM25Retriever,
    semantic: SemanticRetriever,
    bm25_weight: float = 0.5,
    semantic_weight: float = 0.5,
    rrf_k: int = 60,
) -> EnsembleRetriever:
    """
    Combine BM25 and Semantic Retrieval using weighted RRF.

    Both retrievers must implement LangChain BaseRetriever.
    """

    if bm25_weight < 0 or semantic_weight < 0:
        raise ValueError("Retriever weights cannot be negative")

    if bm25_weight + semantic_weight == 0:
        raise ValueError("At least one retriever weight must be positive")

    if rrf_k < 0:
        raise ValueError("rrf_k cannot be negative")

    return EnsembleRetriever(
        retrievers=[bm25, semantic],
        weights=[bm25_weight, semantic_weight],
        c=rrf_k,
        id_key="section_id",
    )


def main():
    # 1. Load the frozen engineering corpus.
    sections = json.loads(
        CATALOG_PATH.read_text(encoding="utf-8")
    )

    # 2. Initialize retrievers with the same fetch_k.
    fetch_k = 20

    bm25 = BM25Retriever(
        sections=sections,
        k=fetch_k,
    )

    semantic = SemanticRetriever(
        k=fetch_k,
    )

    if not semantic.has_index():
        print("Indexing engineering sections...")
        semantic.index(sections)
    else:
        print("Reusing existing vector index...")

    # 3. Create Hybrid Retriever.
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

    # 4. Execute all three retrievers.
    bm25_results = bm25.invoke(query)[:5]
    semantic_results = semantic.invoke(query)[:5]
    hybrid_results = hybrid.invoke(query)[:5]

    print(f"\nQUERY: {query}")

    for name, results in (
        ("BM25", bm25_results),
        ("SEMANTIC", semantic_results),
        ("HYBRID (RRF)", hybrid_results),
    ):
        print(f"\n=== {name} ===")

        for rank, document in enumerate(results, start=1):
            print(
                f"{rank}. "
                f"{document.metadata['section_id']}"
            )


if __name__ == "__main__":
    main()
