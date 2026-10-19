from pathlib import Path
from langchain_community.document_loaders import BSHTMLLoader
from langchain_core.documents import Document

def load_html(file_path: Path) -> list[Document]:
    loader = BSHTMLLoader(str(file_path))
    return loader.load()