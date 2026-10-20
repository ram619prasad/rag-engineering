
from pydantic import ConfigDict
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from bm25_search import BM25Search
from semantic_search import SemanticSearch


class BM25LangChainRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    search_engine: BM25Search
    candidate_k: int = 10

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager,
    ) -> list[Document]:
        results = self.search_engine.search(
            query, top_k=self.candidate_k
        )

        return [
            Document(
                page_content=result["content"],
                metadata={
                    **result["metadata"],
                    "chunk_id": result["chunk_id"],
                },
            )
            for result in results
        ]


class SemanticLangChainRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    search_engine: SemanticSearch
    candidate_k: int = 10

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager,
    ) -> list[Document]:
        results = self.search_engine.search(
            query, top_k=self.candidate_k
        )

        return [
            Document(
                page_content=result["content"],
                metadata={
                    **result["metadata"],
                    "chunk_id": result["chunk_id"],
                },
            )
            for result in results
        ]
