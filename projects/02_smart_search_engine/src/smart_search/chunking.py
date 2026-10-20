
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict, Field

from models import Document, Chunk

def chunk_document(
    document: Document,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        strip_whitespace=True,
    )

    text_chunks = splitter.split_text(document.content.strip())

    return [
        Chunk(
            chunk_id=f"{document.document_id}_chunk_{index:03d}",
            document_id=document.document_id,
            title=document.title,
            content=text,
            category=document.category,
            source=document.source,
            chunk_index=index,
        )
        for index, text in enumerate(text_chunks)
        if text
    ]


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    return [
        chunk
        for document in documents
        for chunk in chunk_document(document, chunk_size, overlap)
    ]


def main() -> None:
    from load_documents import load_documents

    documents = load_documents()
    chunks = chunk_documents(documents)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"First chunk ID: {chunks[0].chunk_id}")
    print(f"First chunk title: {chunks[0].title}")
    print(f"First chunk length: {len(chunks[0].content)} characters")
    print(f"First chunk content: {chunks[0].content}")


if __name__ == "__main__":
    main()
