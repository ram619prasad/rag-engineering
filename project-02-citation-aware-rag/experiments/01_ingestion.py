from datasets import load_dataset

from project_02_citation_aware_rag.ingestion import paper_to_documents

dataset = load_dataset("allenai/qasper", split="train")
paper = dataset[0]

documents = paper_to_documents(paper)

print("Paper ID:", paper["id"])
print("Paper title:", paper["title"])
print("Total documents:", len(documents))

print("\nFirst document:")
print("ID:", documents[0].id)
print("Metadata:", documents[0].metadata)
print("Content preview:", documents[0].page_content[:200])