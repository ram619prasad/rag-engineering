
from pathlib import Path

from advanced_retrieval.loaders.markdown_loader import load_markdown
from advanced_retrieval.loaders.section_extractor import extract_sections


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DOCUMENTS_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "engineering_kb"
    / "documents"
)


def inspect_corpus():
    paths = sorted(DOCUMENTS_DIR.glob("*.md"))
    total_sections = 0

    for path in paths:
        document = load_markdown(str(path))[0]
        sections = extract_sections(document)

        total_sections += len(sections)

        print(
            f"{path.name}: "
            f"{len(sections)} sections"
        )

    # Outside the for loop
    print("\n=== CORPUS SUMMARY ===")
    print(f"Total documents: {len(paths)}")
    print(f"Total sections: {total_sections}")
    print(
        f"Average sections/document: "
        f"{total_sections / len(paths):.1f}"
    )


if __name__ == "__main__":
    inspect_corpus()
