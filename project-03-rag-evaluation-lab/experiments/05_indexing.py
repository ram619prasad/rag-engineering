from project_03_rag_evaluation_lab.dataset import load_qasper
from project_03_rag_evaluation_lab.ingestion import paper_to_documents
from project_03_rag_evaluation_lab.vector_store import create_vector_store
from project_03_rag_evaluation_lab.indexing import index_documents

paper = load_qasper()[0]
documents = paper_to_documents(paper)

vector_store = create_vector_store()

first_run = index_documents(vector_store, documents)
second_run = index_documents(vector_store, documents)

print("Total documents:", len(documents))
print("First run newly indexed:", first_run)
print("Second run newly indexed:", second_run)
print("Collection count:", vector_store._collection.count())