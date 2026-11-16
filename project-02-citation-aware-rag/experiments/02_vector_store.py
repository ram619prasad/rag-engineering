from project_02_citation_aware_rag.vector_store import create_vector_store

vector_store = create_vector_store()

print("Collection:", vector_store._collection.name)
print("Documents indexed:", vector_store._collection.count())