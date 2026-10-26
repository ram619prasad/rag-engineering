
from pathlib import Path

from langchain_community.document_loaders import TextLoader


def load_markdown(path: str):
    file_path = Path(path)

    documents = TextLoader(
        str(file_path),
        encoding="utf-8",
    ).load()

    for document in documents:
        document.metadata["document_id"] = file_path.stem
        document.metadata["format"] = "markdown"

    return documents


def main():
    project_root = Path(__file__).resolve().parents[3]

    path = (
        project_root
        / "datasets/engineering_kb/documents"
        / "kubernetes_resource_management.md"
    )

    documents = load_markdown(str(path))

    print(f"Documents loaded: {len(documents)}")

    document = documents[0]

    print(f"Document ID: {document.metadata['document_id']}")
    print(f"Characters: {len(document.page_content)}")
    print(f"Preview:\n{document.page_content[:500]}")


if __name__ == "__main__":
    main()
