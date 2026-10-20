
from pathlib import Path

import chromadb
import numpy as np

from models import Chunk


class VectorIndex:
    def __init__(
        self,
        collection_name: str = "smart_search",
        persist_directory: str | Path | None = None,
    ) -> None:
        if persist_directory is None:
            project_dir = Path(__file__).resolve().parents[2]
            persist_directory = project_dir / ".chroma"

        self.client = chromadb.PersistentClient(
            path=str(persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required.")

        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError(
                "There must be exactly one embedding per chunk."
            )

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "category": chunk.category,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ],
            embeddings=embeddings.tolist(),
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[dict]:
        if query_embedding.ndim != 1:
            raise ValueError("Query embedding must be a 1D vector.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        matches = []

        for i, chunk_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]

            matches.append(
                {
                    "chunk_id": chunk_id,
                    "content": results["documents"][0][i],
                    "metadata": metadata,
                    "score": 1.0 - float(distance),
                }
            )

        return matches


def main() -> None:
    print("VectorIndex module loaded successfully.")
    print("Next: connect chunk embeddings and query embeddings.")


if __name__ == "__main__":
    main()
