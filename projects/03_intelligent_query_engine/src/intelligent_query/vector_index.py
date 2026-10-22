
from pathlib import Path

import chromadb

from embeddings import EmbeddingGenerator
from models import Chunk


PROJECT_DIR = Path(__file__).resolve().parents[2]
INDEX_DIR = PROJECT_DIR / ".chroma"

COLLECTION_NAME = "scifact_embeddings"
DISTANCE_METRIC = "cosine"


class VectorIndex:
    def __init__(self, embeddings: EmbeddingGenerator) -> None:
        self.embeddings = embeddings

        self.client = chromadb.PersistentClient(
            path=str(INDEX_DIR)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
            metadata={
                "hnsw:space": DISTANCE_METRIC,
                "embedding_model": embeddings.model,
                "embedding_dimensions": embeddings.dimensions,
            },
        )

    def index_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = 100,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # Read existing IDs and metadata.
        existing = self.collection.get(
            include=["metadatas"]
        )

        existing_hashes = {
            chunk_id: metadata.get("content_hash")
            for chunk_id, metadata in zip(
                existing["ids"],
                existing["metadatas"],
            )
        }

        # Select only new or changed chunks.
        pending = [
            chunk
            for chunk in chunks
            if existing_hashes.get(chunk.chunk_id)
            != chunk.content_hash
        ]

        print(f"Total input chunks: {len(chunks)}")
        print(f"New or changed chunks: {len(pending)}")

        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]

            vectors = self.embeddings.embed_chunks(batch)

            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                embeddings=vectors.tolist(),
                metadatas=[
                    {
                        "document_id": chunk.document_id,
                        "title": chunk.title,
                        "chunk_index": chunk.chunk_index,
                        "source": chunk.source,
                        "content_hash": chunk.content_hash,
                    }
                    for chunk in batch
                ],
            )

            print(
                f"Indexed {start + len(batch)}"
                f"/{len(pending)} pending chunks"
            )

        print(
            f"Collection size: {self.collection.count()}"
        )


def main() -> None:
    from chunking import chunk_documents
    from dataset import load_scifact

    corpus, _, _ = load_scifact()
    chunks = chunk_documents(corpus)

    generator = EmbeddingGenerator(batch_size=100)
    index = VectorIndex(generator)

    # Estimate cost before indexing
    tokens, estimated_cost = generator.estimate_cost(chunks)

    print(f"Total chunks: {len(chunks)}")
    print(f"Estimated tokens: {tokens:,}")
    print(f"Estimated full-corpus cost: ${estimated_cost:.4f}")

    # Incremental indexing:
    # Only new or modified chunks generate embeddings
    index.index_chunks(chunks, batch_size=100)


if __name__ == "__main__":
    main()