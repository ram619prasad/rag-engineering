from pathlib import Path

from document_ingestion.loaders.markdown_loader import load_markdown
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
    file_path = (
        project_root / "datasets" / "sample" / "engineering_notes.md"
    )

    documents = load_markdown(file_path)
    print(f"Number of documents: {len(documents)}")

    for document in documents:
        print("\n--- Metadata ---")
        print(document.metadata)

        print("\n--- Content preview ---")
        print(document.page_content[:200])

        errors = validate_document(document)

        if errors:
            print("Invalid document:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("Document is valid")


if __name__ == "__main__":
    main()