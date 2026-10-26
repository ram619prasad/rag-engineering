
from langchain_text_splitters import MarkdownHeaderTextSplitter

from advanced_retrieval.loaders.markdown_loader import load_markdown


HEADERS = [
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
    ("#####", "h5"),
]


def extract_sections(document):
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS,
        strip_headers=False,
    )

    sections = splitter.split_text(document.page_content)

    for index, section in enumerate(sections):
        section.metadata.update(document.metadata)
        section.metadata["section_id"] = (
            f"{document.metadata['document_id']}:s{index:03d}"
        )

    return sections


def main():
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    path = (
        project_root
        / "datasets/engineering_kb/documents"
        / "kubernetes_resource_management.md"
    )

    document = load_markdown(str(path))[0]
    sections = extract_sections(document)

    print(f"Total sections: {len(sections)}")

    for section in sections[:8]:
        print("\n---")
        print("ID:", section.metadata["section_id"])
        print("Headings:", {
            k: v
            for k, v in section.metadata.items()
            if k.startswith("h")
        })
        print("Characters:", len(section.page_content))
        print("Preview:", section.page_content[:120])


if __name__ == "__main__":
    main()
