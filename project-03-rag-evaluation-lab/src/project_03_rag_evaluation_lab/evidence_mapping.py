from langchain_core.documents import Document


def map_gold_evidence(
    gold_evidence: list[str],
    documents: list[Document],
) -> tuple[set[str], list[str]]:
    paragraph_lookup = {
        doc.page_content.strip(): doc.id
        for doc in documents
        if doc is not None
    }
    
    matched_ids: set[str] = set()
    unmatched_evidence: list[str] = []

    for evidence in gold_evidence:
        document_id = paragraph_lookup.get(evidence.strip())

        if document_id is not None:
            matched_ids.add(document_id)
        else:
            unmatched_evidence.append(evidence)
            
    return(matched_ids, unmatched_evidence)