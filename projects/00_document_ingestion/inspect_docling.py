from pathlib import Path
from docling.document_converter import DocumentConverter

pdf_path = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "sample"
    / "tech_doc.pdf"
)

converter = DocumentConverter()
result = converter.convert(pdf_path)
print(result.document.export_to_markdown())