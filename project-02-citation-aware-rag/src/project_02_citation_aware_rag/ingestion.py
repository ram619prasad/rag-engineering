from langchain_core.documents import Document

def paper_to_documents(paper: dict) -> list[Document]:
    documents: list[Document] = []
    paper_id = paper["id"]
    
    for section_index, (section, paragraphs) in enumerate(
        zip(
           paper["full_text"]["section_name"],
           paper["full_text"]["paragraphs"] 
        )
    ):
        for paragraph_index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            
            document = Document(
                id=f"{paper_id}:{section_index}:{paragraph_index}",
                page_content=paragraph,
                metadata={
                    "paper_id": paper_id,
                    "title": paper["title"],
                    "section": section,
                    "section_index": section_index,
                    "paragraph_index": paragraph_index,
                }    
            )
            
            documents.append(document)
            
    return documents

