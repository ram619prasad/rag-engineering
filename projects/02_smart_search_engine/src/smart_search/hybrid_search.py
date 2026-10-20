
from langchain_classic.retrievers import EnsembleRetriever

from bm25_search import BM25Search
from chunking import chunk_documents
from load_documents import load_documents
from retriever import (
    BM25LangChainRetriever,
    SemanticLangChainRetriever,
)
from semantic_search import SemanticSearch


class HybridSearch:
    def __init__(
        self,
        candidate_k: int = 10,
        rrf_k: int = 60,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero.")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        documents = load_documents()
        chunks = chunk_documents(documents)
        bm25 = BM25Search(chunks)

        semantic = SemanticSearch()
        semantic.index_documents()

        self.retriever = EnsembleRetriever(
            retrievers=[
                BM25LangChainRetriever(
                    search_engine=bm25,
                    candidate_k=candidate_k,
                ),
                SemanticLangChainRetriever(
                    search_engine=semantic,
                    candidate_k=candidate_k,
                ),
            ],
            weights=[0.5, 0.5],
            c=rrf_k,
            id_key="chunk_id",
        )

    def search(self, query: str, top_k: int = 3):
        if not query.strip():
            raise ValueError("query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        return self.retriever.invoke(query)[:top_k]


def main() -> None:
    search_engine = HybridSearch()

    queries = [
        "ERR_CONN_RESET_1042",
        "HTTP 429 Too Many Requests",
        "How can I reduce Kubernetes infrastructure costs?",
        "How can I spend less on my cluster?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")

        for rank, document in enumerate(
            search_engine.search(query), start=1
        ):
            print(
                f"  {rank}. {document.metadata['title']} | "
                f"{document.metadata['chunk_id']}"
            )


if __name__ == "__main__":
    main()
