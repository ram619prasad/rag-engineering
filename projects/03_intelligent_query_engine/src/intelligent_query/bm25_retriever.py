
import re

from rank_bm25 import BM25Okapi
import numpy as np
from models import RetrievalResult



def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    def __init__(self, corpus):
        self.documents = list(corpus)

        # Include title and abstract in the searchable text.
        tokenized_corpus = [
            tokenize(f"{doc['title']} {doc['text']}")
            for doc in self.documents
        ]

        # Build the BM25 index once.
        self.index = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        scores = self.index.get_scores(tokenize(query))

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        return [
            RetrievalResult(
                document_id=str(self.documents[i]["id"]),
                title=self.documents[i]["title"],
                text=self.documents[i]["text"],
                score=float(scores[i]),
                retrieval_method="bm25",
            )
            for i in ranked_indices
        ]

