from pathlib import Path

from document_ingestion.loaders.pdf_loader import load_pdf

file_path = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "sample"
    / "sample-local-pdf.pdf"
)

documents = load_pdf(file_path)

print(f"Number of page documents: {len(documents)}")

for document in documents:
    print("\n--- Metadata ---")
    print(document.metadata)

    print("\n--- Content preview ---")
    print(document.page_content[:300])