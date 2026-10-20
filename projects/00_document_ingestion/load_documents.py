from pathlib import Path

from document_ingestion.loaders.document_loader import load_document
from document_ingestion.validation.document_validator import validate_document


def find_project_root(start: Path) -> Path:
    for directory in (start, *start.parents):
        if (
            (directory / "pyproject.toml").exists()
            and (directory / "datasets").is_dir()
            and (directory / "projects").is_dir()
        ):
            return directory

    raise FileNotFoundError("Could not find repository root")


def main() -> None:
    project_root = find_project_root(Path(__file__).resolve().parent)
    engineering_notes_file_path = (
        project_root / "datasets" / "sample" / "engineering_notes.md"
    )
    meeting_notes_file_path = (
        project_root / "datasets" / "sample" / "meeting_notes.md"
    )
    missing_notes_file_path = (
        project_root / "datasets" / "sample" / "missing_notes.md"
    )
    
    docs_file_path = (
        project_root / "datasets" / "sample" / "meeting_notes.docx"
    )

    meeting_notes_text_path = (
        project_root / "datasets" / "sample" / "meeting_notes.txt"
    )

    html_file_path = (
        project_root / "datasets" / "sample" / "rag.html"
    )

    pdf_file_path = (
        project_root / "datasets" / "sample" / "sample.pdf"
    )

    unknown_file_path = (
        project_root / "datasets" / "sample" / "rag.uvw"
    )

    files_to_load = [
        engineering_notes_file_path, meeting_notes_file_path,
        missing_notes_file_path, meeting_notes_text_path,
        docs_file_path, html_file_path,
        pdf_file_path, unknown_file_path
    ]
    # files_to_load = [pdf_file_path]

    successful_files = []
    failed_files = []
    invalid_documents = []
    valid_documents = []


    for path in files_to_load:
        try:
            documents = load_document(path)
            print(f"Number of documents: {len(documents)}")
            successful_files.append(path)

            for document in documents:
                print("\n--- Metadata ---")
                print(document.metadata)

                print("\n--- Content preview ---")
                print(document.page_content[:200])

                errors = validate_document(document)

                if errors:
                    print("Invalid document:")
                    invalid_documents.append(document)
                    for error in errors:
                        print(f"  - {error}")
                else:
                    valid_documents.append(document)
                    print("Document is valid")
        except Exception as e:
            failed_files.append(path)
            print(f"Failed to load the markdown at {path}: {e}")

    
    print("\n--- Ingestion Summary ---")
    print(f"Total files: {len(files_to_load)}")
    print(f"Successfully loaded files: {len(successful_files)}")
    print(f"Failed files: {len(failed_files)}")
    print(f"Valid documents: {len(valid_documents)}")
    print(f"Invalid documents: {len(invalid_documents)}")


if __name__ == "__main__":
    main()