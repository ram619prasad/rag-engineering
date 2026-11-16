from datasets import load_dataset
from langchain_core.documents import Document


def paper_to_documents(paper: dict) -> list[Document]:
    documents: list[Document] = []
    
    full_text = paper["full_text"]
    sections = full_text["section_name"]
    section_paragraphs = full_text["paragraphs"]

    for section_index, section_name in enumerate(sections):
        paragraphs = section_paragraphs[section_index]
        for paragraph_index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            
            document_id = (
                f"{paper['id']}:{section_index}:{paragraph_index}"
            )
            
            metadata = {
                "paper_id": paper["id"],
                "title": paper["title"],
                "section": section_name,
                "section_index": section_index,
                "paragraph_index": paragraph_index,
            }
            
            documents.append(
                Document(
                    id=document_id,
                    page_content=paragraph,
                    metadata=metadata,
                )
            )

    return documents


if __name__ == "__main__":
    dataset = load_dataset("allenai/qasper", split="train")
    paper = dataset[0]
    
    documents = paper_to_documents(paper)

    print("Total documents:", len(documents))
    print("First document metadata:", documents[0].metadata)
    print("First document text:", documents[0].page_content[:150])
    
    
    expected_count = sum(
        1
        for section in paper["full_text"]["paragraphs"]
        for paragraph in section
        if paragraph.strip()
    )

    assert len(documents) == expected_count

    gold_matches = [
        doc
        for doc in documents
        if doc.metadata["section_index"] == 4
        and doc.metadata["paragraph_index"] == 1
    ]

    assert len(gold_matches) == 1
    gold_evidence = paper["qas"]["answers"][0]["answer"][0]["evidence"][0]
    assert gold_matches[0].page_content == gold_evidence

    print("Expected document count:", expected_count)
    print("Gold evidence preserved:", True)
    print("Ingestion validation passed!")