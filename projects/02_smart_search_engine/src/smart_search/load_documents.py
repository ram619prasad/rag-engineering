import json
from pathlib import Path

from models import Document

def load_documents() -> list[Document]:
    project_dir = Path(__file__).resolve().parents[2]
    dataset_path = project_dir / "datasets" / "documents.json"

    with dataset_path.open("r", encoding="utf-8") as file:
        raw_documents = json.load(file)

    documents = [
        Document.model_validate(item)
        for item in raw_documents
    ]

    document_ids = [document.document_id for document in documents]

    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Document IDs must be unique.")

    return documents

def main() -> None:
    documents = load_documents()

    print(f"Loaded {len(documents)} documents")
    print(f"First document: {documents[0].title}")
    print(f"Category: {documents[0].category}")


if __name__ == "__main__":
    main()