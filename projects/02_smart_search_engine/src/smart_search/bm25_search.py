
import re

from rank_bm25 import BM25Okapi

from chunking import chunk_documents
from load_documents import load_documents
from models import Chunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Search:
    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("chunks cannot be empty.")

        self.chunks = chunks
        tokenized_corpus = [
            tokenize(chunk.content) for chunk in chunks
        ]
        self.index = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not query.strip():
            raise ValueError("query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        scores = self.index.get_scores(tokenize(query))

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []
        for i in ranked_indices:
            if scores[i] <= 0:
                continue

            chunk = self.chunks[i]
            results.append({
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": {
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "category": chunk.category,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                },
                "score": float(scores[i]),
            })

            if len(results) == top_k:
                break

        return results


def main() -> None:
    documents = load_documents()
    chunks = chunk_documents(documents)
    search_engine = BM25Search(chunks)

    queries = [
        "ERR_CONN_RESET_1042",
        "HTTP 429 Too Many Requests",
        "How can I reduce Kubernetes infrastructure costs?",
        "How can I spend less on my cluster?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")

        for result in search_engine.search(query):
            print(
                f"  {result['score']:.3f} | "
                f"{result['metadata']['title']} | "
                f"{result['chunk_id']}"
            )


if __name__ == "__main__":
    main()
