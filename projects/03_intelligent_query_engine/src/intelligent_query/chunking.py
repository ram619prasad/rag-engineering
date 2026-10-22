import hashlib

from datasets import Dataset

from models import Chunk

def create_chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}_chunk_{chunk_index:03d}"


def chunk_documents(corpus: Dataset) -> list[Chunk]:
    chunks = []

    for document in corpus:
        document_id = str(document["id"])
        text = f"{document['title']}\n{document['text']}"

        # One chunk per document for our current SciFact baseline
        chunk_index = 0

        chunks.append(
            Chunk(
                chunk_id=create_chunk_id(document_id, chunk_index),
                document_id=document_id,
                title=document["title"],
                text=text,
                chunk_index=chunk_index,
                source="scifact",
                content_hash=hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
            )
        )

    return chunks