from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


def load_markdown(file_path: Path) -> list[Document]:
    loader = TextLoader(str(file_path), encoding="utf-8")
    return loader.load()
