
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from pydantic import PrivateAttr

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun


ROOT = Path(__file__).resolve().parents[3]

CATALOG_PATH = (
    ROOT / "datasets" / "engineering_kb" / "section_catalog.json"
)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever(BaseRetriever):
    sections: list[dict]
    k: int = 20

    # Private internal state, not a Pydantic model field.
    _bm25: BM25Okapi = PrivateAttr()

    def model_post_init(self, __context) -> None:
        tokenized_corpus = [
            tokenize(section["text"])
            for section in self.sections
        ]

        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, k: int = 5) -> list[dict]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked_indices = np.argsort(scores)[::-1][:k]

        return [
            {
                "section_id": self.sections[i]["section_id"],
                "document_id": self.sections[i]["document_id"],
                "heading_path": self.sections[i]["heading_path"],
                "score": float(scores[i]),
                "text": self.sections[i]["text"],
            }
            for i in ranked_indices
        ]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        results = self.search(query, k=self.k)

        return [
            Document(
                id=result["section_id"],
                page_content=result["text"],
                metadata={
                    "section_id": result["section_id"],
                    "document_id": result["document_id"],
                    "heading_path": " > ".join(result["heading_path"]),
                    "bm25_score": result["score"],
                },
            )
            for result in results
        ]


def main():
    sections = json.loads(
        CATALOG_PATH.read_text(encoding="utf-8")
    )

    # Pydantic models use keyword arguments.
    retriever = BM25Retriever(sections=sections, k=5)

    query = "How does Kubernetes HPA scale pods based on CPU utilization?"

    # Original BM25 interface — includes scores.
    results = retriever.search(query, k=5)

    print("\n=== BM25 SEARCH ===")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['section_id']} "
            f"(score={result['score']:.3f})"
        )

    # LangChain interface — compatible with EnsembleRetriever.
    documents = retriever.invoke(query)

    for rank, document in enumerate(documents, start=1):
        print(
            f"{rank}. {document.metadata['section_id']}"
        )


if __name__ == "__main__":
    main()
