
from typing import Any
from dataset import load_scifact
import random

from dotenv import load_dotenv
from pydantic import PrivateAttr

from langchain_classic.retrievers.multi_query import (
    MultiQueryRetriever,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
)
from langchain_openai import ChatOpenAI

from embeddings import EmbeddingGenerator
from vector_index import VectorIndex
from semantic_retriever import SemanticRetriever

import logging

logging.basicConfig(level=logging.WARNING)

logging.getLogger(
    "langchain_classic.retrievers.multi_query"
).setLevel(logging.INFO)


load_dotenv()


class SemanticRetrieverAdapter(BaseRetriever):
    """Expose our existing retriever to LangChain."""

    _semantic: Any = PrivateAttr()
    _top_k: int = PrivateAttr()

    def __init__(
        self,
        semantic: SemanticRetriever,
        top_k: int = 5,
    ):
        super().__init__()
        self._semantic = semantic
        self._top_k = top_k

    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:

        results = self._semantic.search(
            query,
            top_k=self._top_k,
        )

        return [
            Document(
                page_content=result.text,
                metadata={
                    "document_id": result.document_id,
                    "title": result.title,
                },
                id=result.document_id,
            )
            for result in results
        ]


def main() -> None:
    semantic = SemanticRetriever(
        VectorIndex(EmbeddingGenerator())
    )

    adapter = SemanticRetrieverAdapter(
        semantic=semantic,
        top_k=5,
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    multi_query = MultiQueryRetriever.from_llm(
        retriever=adapter,
        llm=llm,
        include_original=True,
    )

    _, queries, qrels = load_scifact()

    sample_query = random.choice(queries)

    query_id = str(sample_query["id"])
    query = sample_query["text"]
    
    documents = multi_query.invoke(query)

    relevant_ids = {
        str(row["corpus-id"])
        for row in qrels
        if str(row["query-id"]) == query_id
        and int(row["score"]) > 0
    }

    retrieved_ids = {
        doc.metadata["document_id"]
        for doc in documents
    }

    found_relevant = relevant_ids & retrieved_ids

    print(f"\nRelevant documents: {relevant_ids}")
    print(f"Relevant documents found: {found_relevant}")

    print(
        f"Relevant evidence coverage: "
        f"{len(found_relevant)}/{len(relevant_ids)}"
    )


if __name__ == "__main__":
    main()
