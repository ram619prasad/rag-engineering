
from langchain_core.documents import Document


def paper_to_documents(paper: dict) -> list[Document]:
    documents: list[Document] = []
    paper_id = paper["id"]
    title = paper["title"]
    full_text = paper["full_text"]

    for section_index, (section, paragraphs) in enumerate(
        zip(
            full_text["section_name"],
            full_text["paragraphs"],
        )
    ):
        for paragraph_index, paragraph in enumerate(paragraphs):
            if not paragraph or not paragraph.strip():
                continue

            # Stable ID: paper_id:section_index:paragraph_index
            document_id = (
                f"{paper_id}:{section_index}:{paragraph_index}"
            )

            document = Document(
                id=document_id,
                page_content=paragraph,
                metadata={
                    "paper_id": paper_id,
                    "title": title,
                    "section": section,
                    "section_index": section_index,
                    "paragraph_index": paragraph_index,
                },
            )

            documents.append(document)

    return documents
