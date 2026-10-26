
import json

from advanced_retrieval.loaders.corpus_inspector import DOCUMENTS_DIR
from advanced_retrieval.loaders.markdown_loader import load_markdown
from advanced_retrieval.loaders.section_extractor import extract_sections


def build_catalog():
    catalog = []

    for path in sorted(DOCUMENTS_DIR.glob("*.md")):
        document = load_markdown(str(path))[0]
        sections = extract_sections(document)

        for section in sections:
            catalog.append({
                "section_id": section.metadata["section_id"],
                "document_id": section.metadata["document_id"],
                "heading_path": [
                    section.metadata[key]
                    for key in ("h2", "h3", "h4", "h5")
                    if key in section.metadata
                ],
                "text": section.page_content,
            })

    output = DOCUMENTS_DIR.parent / "section_catalog.json"

    output.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Documents: {len(set(s['document_id'] for s in catalog))}")
    print(f"Sections: {len(catalog)}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    build_catalog()
