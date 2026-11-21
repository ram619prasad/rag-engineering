from project_03_rag_evaluation_lab.dataset import load_qasper
from project_03_rag_evaluation_lab.vector_store import create_vector_store

dataset = load_qasper()
paper = dataset[0]
vector_store = create_vector_store()

print("Collection:", vector_store._collection.name)
print("Indexed documents:", vector_store._collection.count())