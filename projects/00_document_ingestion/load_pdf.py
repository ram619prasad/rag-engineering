from pathlib import Path
import pymupdf
from pathlib import Path

from document_ingestion.loaders.pdf_loader import load_pdf

pdf_path = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "sample"
    / "tech_doc.pdf"
)


def main() -> None:
    # project_root = Path(__file__).resolve().parents[2]
    # pdf_path = project_root / "datasets" / "sample" / "tech_doc.pdf"

    documents = load_pdf(pdf_path)

    for document in documents:
        print(f"\n{'=' * 20} PAGE {document.metadata['page'] + 1} {'=' * 20}")
        print(document.page_content)


if __name__ == "__main__":
    main()


# with pymupdf.open(pdf_path) as pdf:
#     for page_number, page in enumerate(pdf, start=1):
#         text = page.get_text().strip()
#         images = page.get_images(full=True)

#         print(
#             f"Page {page_number}: "
#             f"text_chars={len(text)}, "
#             f"images={len(images)}"
#         )


# with pymupdf.open(pdf_path) as pdf:
#     page = pdf[0]  # Page 1

#     for image_number, image_info in enumerate(
#         page.get_images(full=True), start=1
#     ):
#         xref = image_info[0]
#         image = pymupdf.Pixmap(pdf, xref)

#         print(
#             f"Image {image_number}: "
#             f"{image.width} x {image.height} pixels"
#         )


with pymupdf.open(pdf_path) as pdf:
    page = pdf[2]  # Page 3; indexing starts at 0

    output_dir = pdf_path.parent / "extracted_images"
    output_dir.mkdir(exist_ok=True)

    for image_number, image_info in enumerate(
        page.get_images(full=True), start=1
    ):
        xref = image_info[0]
        image_data = pdf.extract_image(xref)

        image_path = output_dir / (
            f"page3_image_{image_number}.{image_data['ext']}"
        )

        image_path.write_bytes(image_data["image"])
        print(f"Saved: {image_path}")