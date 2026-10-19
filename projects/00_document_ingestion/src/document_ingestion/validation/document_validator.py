from langchain_core.documents import Document


def validate_document(document: Document) -> list[str]:
    errors = []

    if len(document.page_content.strip()) < 20:
        errors.append("Document content is empty or too short")

    if not document.metadata.get("source"):
        errors.append("Missing source metadata")

    return errors
