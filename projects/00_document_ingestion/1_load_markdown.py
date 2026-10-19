from pathlib import Path

from langchain_community.document_loaders import TextLoader


def find_project_root(start: Path) -> Path:
    for directory in (start, *start.parents):
        if (directory / "pyproject.toml").exists():
            return directory

    raise FileNotFoundError("Could not find project root")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)

FILE_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "sample"
    / "engineering_notes.md"
)

def main() -> None:
    loader = TextLoader(str(FILE_PATH), encoding="utf-8")
    documents = loader.load()

    print(f"Number of documents: {len(documents)}")

    for document in documents:
        print("\n--- Metadata ---")
        print(document.metadata)

        print("\n--- Content preview ---")
        print(document.page_content[:200])


if __name__ == "__main__":
    main()
