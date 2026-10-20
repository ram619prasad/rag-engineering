
from chunking import chunk_documents
from embeddings import EmbeddingGenerator
from load_documents import load_documents
from vector_index import VectorIndex


class SemanticSearch:
    def __init__(self) -> None:
        self.embedding_generator = EmbeddingGenerator()
        self.vector_index = VectorIndex()

    def index_documents(self) -> None:
        documents = load_documents()
        chunks = chunk_documents(documents)

        existing_ids = set(
            self.vector_index.collection.get(
                include=[]
            )["ids"]
        )
        expected_ids = {chunk.chunk_id for chunk in chunks}

        if expected_ids.issubset(existing_ids):
            print(f"Reusing existing index ({len(expected_ids)} chunks)")
            return

        embeddings = self.embedding_generator.embed_chunks(chunks)
        self.vector_index.add_chunks(chunks, embeddings)

        print(f"Indexed {len(chunks)} chunks in ChromaDB")
    

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_embedding = self.embedding_generator.embed_query(query)
        return self.vector_index.search(query_embedding, top_k=top_k)


def main() -> None:
    search_engine = SemanticSearch()
    search_engine.index_documents()

    queries = [
        "ERR_CONN_RESET_1042",
        "HTTP 429 Too Many Requests",
        "How can I reduce Kubernetes infrastructure costs?",
        "How can I spend less on my cluster?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")

        results = search_engine.search(query, top_k=3)

        for result in results:
            print(
                f"  {result['score']:.3f} | "
                f"{result['metadata']['title']} | "
                f"{result['chunk_id']}"
            )


if __name__ == "__main__":
    main()
