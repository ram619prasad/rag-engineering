
from embeddings import EmbeddingGenerator
from models import RetrievalResult
from vector_index import VectorIndex


class SemanticRetriever:
    def __init__(self, index: VectorIndex) -> None:
        self.index = index
        self.embeddings = index.embeddings

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if self.index.collection.count() == 0:
            raise ValueError("Vector index is empty")

        # Only embed the query, not the corpus.
        query_vector = self.embeddings.embed_query(query)

        if query_vector.shape != (
            self.embeddings.dimensions,
        ):
            raise ValueError(
                f"Unexpected query embedding shape: "
                f"{query_vector.shape}"
            )

        results = self.index.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(
                top_k,
                self.index.collection.count(),
            ),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        retrieved = []

        for chunk_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append(
                RetrievalResult(
                    document_id=str(metadata["document_id"]),
                    title=str(metadata["title"]),
                    text=text,
                    score=1.0 - float(distance),
                    retrieval_method="semantic",
                )
            )

        return retrieved


def main() -> None:
    generator = EmbeddingGenerator()
    index = VectorIndex(generator)
    retriever = SemanticRetriever(index)

    query = "0-dimensional biomaterials show inductive properties."
    target_document_id = "31715818"

    # Retrieve the top 100 candidates
    results = retriever.search(query, top_k=100)

    for rank, result in enumerate(results, start=1):
        if result.document_id == target_document_id:
            print(f"Relevant document rank: {rank}")
            print(f"Similarity: {result.score:.4f}")
            break
    else:
        print("Relevant document not found in top 100")


if __name__ == "__main__":
    main()
