
import json
from pathlib import Path

from dotenv import load_dotenv
from pydantic import PrivateAttr

from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings


load_dotenv()


ROOT = Path(__file__).resolve().parents[3]

CATALOG_PATH = (
    ROOT
    / "datasets"
    / "engineering_kb"
    / "section_catalog.json"
)

CHROMA_DIR = (
    ROOT
    / "data"
    / "chroma"
    / "semantic_baseline"
)


class SemanticRetriever(BaseRetriever):
    # Configurable retriever parameters
    k: int = 5

    collection_name: str = "engineering_semantic_baseline"
    persist_directory: Path = CHROMA_DIR
    embedding_model: str = "text-embedding-3-small"

    # Internal state managed by Pydantic
    _embeddings: OpenAIEmbeddings = PrivateAttr()
    _vectorstore: Chroma = PrivateAttr()

    def model_post_init(self, __context) -> None:
        """Initialize embeddings and the persistent Chroma store."""

        self._embeddings = OpenAIEmbeddings(
            model=self.embedding_model
        )

        self._vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(self.persist_directory),
        )

    @property
    def vectorstore(self) -> Chroma:
        """Expose the vector store for reuse when needed."""
        return self._vectorstore

    def index(self, sections: list[dict]) -> None:
        """Insert or update sections using stable IDs."""

        documents = [
            Document(
                id=section["section_id"],
                page_content=section["text"],
                metadata={
                    "section_id": section["section_id"],
                    "document_id": section["document_id"],
                    "heading_path": " > ".join(
                        section["heading_path"]
                    ),
                },
            )
            for section in sections
        ]

        if documents:
            self._vectorstore.add_documents(
                documents=documents
            )

    def has_index(self) -> bool:
        """Check whether the collection contains any documents."""

        results = self._vectorstore.get(
            limit=1,
            include=[],
        )

        return bool(results["ids"])

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[dict]:
        """Search with Chroma distances for debugging."""

        results = self._vectorstore.similarity_search_with_score(
            query,
            k=k,
        )

        return [
            {
                "section_id": document.metadata["section_id"],
                "document_id": document.metadata["document_id"],
                "heading_path": document.metadata["heading_path"],
                "score": float(distance),
                "text": document.page_content,
            }
            for document, distance in results
        ]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """LangChain-compatible retrieval for EnsembleRetriever."""

        return self._vectorstore.similarity_search(
            query,
            k=self.k,
        )


def main():
    sections = json.loads(
        CATALOG_PATH.read_text(encoding="utf-8")
    )

    retriever = SemanticRetriever(k=5)

    if not retriever.has_index():
        print("Indexing engineering sections...")
        retriever.index(sections)
    else:
        print("Reusing existing vector index...")

    query = (
        "How does Kubernetes automatically increase "
        "replicas when CPU usage is high?"
    )

    # Original interface: return distances.
    results = retriever.search(query, k=5)

    print(f"\nQuery: {query}")

    print("\n=== SEMANTIC SEARCH ===")

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['section_id']} "
            f"(distance={result['score']:.4f})"
        )

    # LangChain interface: return Document objects.
    documents = retriever.invoke(query)

    print("\n=== LANGCHAIN INVOKE ===")

    for rank, document in enumerate(documents, start=1):
        print(
            f"{rank}. {document.metadata['section_id']}"
        )


if __name__ == "__main__":
    main()
