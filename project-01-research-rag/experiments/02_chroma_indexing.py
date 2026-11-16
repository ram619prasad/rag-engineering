from datasets import load_dataset

from project_01_research_rag.ingestion import paper_to_documents
from project_01_research_rag.vector_store import create_vector_store
from project_01_research_rag.indexing import index_documents

dataset = load_dataset("allenai/qasper", split="train")
# paper = dataset[0]
# documents = paper_to_documents(paper)
vector_store = create_vector_store()

for paper in dataset.select(range(10)):
    paper = dict(paper)
    documents = paper_to_documents(paper)
    added = index_documents(vector_store, documents)

    print(f"Paper {paper['id']}: {added} new documents")