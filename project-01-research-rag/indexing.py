from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "qasper_research"

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma(
    persist_directory=str(DATA_DIR),
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model
)

print("Collection count:", vector_store._collection.count())

question = "What is the seed lexicon?"

vector = embedding_model.embed_query(question)

print("Embedding dimension:", len(vector))
print("First five values:", vector[:5])