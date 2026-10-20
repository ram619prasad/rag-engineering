from pathlib import Path
from langchain_core.documents import Document

from document_ingestion.loaders.text_loader import load_text
from document_ingestion.loaders.markdown_loader import load_markdown
from document_ingestion.loaders.html_loader import load_html
from document_ingestion.loaders.docs_loader import load_docx
from document_ingestion.loaders.pdf_loader import load_pdf


def load_document(file_path: Path) -> list[Document]:
    extension = file_path.suffix.lower()

    if extension == ".txt":
        return load_text(file_path)
    elif extension in {".html", ".htm"}:
        return load_html(file_path)
    elif extension == ".md":
        return load_markdown(file_path)
    elif extension == ".docx":
        return load_docx(file_path)
    elif extension == ".pdf":
        return load_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")
