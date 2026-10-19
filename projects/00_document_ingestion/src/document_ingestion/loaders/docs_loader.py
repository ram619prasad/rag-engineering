from pathlib import Path
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document

def load_docx(file_path: Path) -> list[Document]:
    loader = Docx2txtLoader(str(file_path))
    return loader.load()