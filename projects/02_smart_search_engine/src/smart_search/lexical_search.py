from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from chunking import chunk_documents
from load_documents import load_documents
from models import Chunk


class LexicalSearch:
    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required.")

        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            strip_accents="unicode",
        )

        self.chunk_vectors = self.vectorizer.fit_transform(
            chunk.content for chunk in chunks
        )

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.chunk_vectors).flatten()

        ranked_indices = scores.argsort()[::-1][:top_k]

        return [
            {
                "chunk": self.chunks[index],
                "score": float(scores[index]),
            }
            for index in ranked_indices
            if scores[index] > 0
        ]


def main() -> None:
    documents = load_documents()
    chunks = chunk_documents(documents)
    search_engine = LexicalSearch(chunks)

    queries = [
        "ERR_CONN_RESET_1042",
        "HTTP 429 Too Many Requests",
        "How can I reduce Kubernetes infrastructure costs?",
        "How can I spend less on my cluster?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")

        results = search_engine.search(query, top_k=3)

        for result in results:
            chunk = result["chunk"]
            score = result["score"]
            print(f"  {score:.3f} | {chunk.title} | {chunk.chunk_id}")


if __name__ == "__main__":
    main()
