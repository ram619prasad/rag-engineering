from datasets import load_dataset

from project_02_citation_aware_rag.ingestion import paper_to_documents
from project_02_citation_aware_rag.vector_store import create_vector_store
from project_02_citation_aware_rag.indexing import index_documents

dataset = load_dataset("allenai/qasper", split="train")
paper = dataset[0]

documents = paper_to_documents(paper)
vector_store = create_vector_store()
print("Before indexing:", vector_store._collection.count())

first_count = index_documents(vector_store, documents)
print("First run newly indexed:", first_count)

second_count = index_documents(vector_store, documents)
print("Second run newly indexed:", second_count)

print("After indexing:", vector_store._collection.count())