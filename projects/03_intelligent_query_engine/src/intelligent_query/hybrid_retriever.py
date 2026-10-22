
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from bm25_retriever import BM25Retriever
from semantic_retriever import SemanticRetriever
from models import RetrievalResult


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Retriever,
        semantic: SemanticRetriever,
        candidate_k: int = 50,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")

        def to_documents(
            results: list[RetrievalResult],
        ) -> list[Document]:
            return [
                Document(
                    page_content=result.text,
                    metadata={
                        "document_id": result.document_id,
                        "title": result.title,
                    },
                )
                for result in results
            ]

        bm25_runnable = RunnableLambda(
            lambda query: to_documents([
                result
                for result in bm25.search(query, top_k=candidate_k)
                if result.score > 0
            ])
        )

        semantic_runnable = RunnableLambda(
            lambda query: to_documents(
                semantic.search(query, top_k=candidate_k)
            )
        )

        self.ensemble = EnsembleRetriever(
            retrievers=[bm25_runnable, semantic_runnable],
            weights=[0.5, 0.5],
            c=60,
            id_key="document_id",
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        documents = self.ensemble.invoke(query)[:top_k]

        return [
            RetrievalResult(
                document_id=doc.metadata["document_id"],
                title=doc.metadata["title"],
                text=doc.page_content,
                score=None,
                retrieval_method="hybrid",
            )
            for doc in documents
        ]


def main() -> None:
    from dataset import load_scifact
    from embeddings import EmbeddingGenerator
    from vector_index import VectorIndex

    corpus, _, _ = load_scifact()

    bm25 = BM25Retriever(corpus)

    generator = EmbeddingGenerator()
    semantic = SemanticRetriever(VectorIndex(generator))

    hybrid = HybridRetriever(
        bm25=bm25,
        semantic=semantic,
        candidate_k=50,
    )

    query = "0-dimensional biomaterials show inductive properties."
    relevant_id = "31715818"

    results = hybrid.search(query, top_k=5)

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result.title}\n"
            f"   Document ID: {result.document_id}\n"
            f"   Relevant: {result.document_id == relevant_id}\n"
        )


if __name__ == "__main__":
    main()