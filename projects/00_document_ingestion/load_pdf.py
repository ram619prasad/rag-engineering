from pathlib import Path
import pymupdf

from document_ingestion.loaders.pdf_loader import load_pdf

file_path = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "sample"
    / "tech_doc.pdf"
)

# documents = load_pdf(file_path)

# print(f"Number of page documents: {len(documents)}")

# for document in documents:
#     print("\n--- Metadata ---")
#     print(document.metadata)

#     print("\n--- Content preview ---")
#     print(document.page_content[:300])

from pathlib import Path

import pymupdf

pdf_path = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "sample"
    / "tech_doc.pdf"
)

with pymupdf.open(pdf_path) as pdf:
    for page_number, page in enumerate(pdf):
        for table in page.find_tables().tables:
            print("Rows:", table.row_count)
            print("Columns:", table.col_count)
            print("Column names/positions:", table.header.names)
            print("Cells:", table.cells)