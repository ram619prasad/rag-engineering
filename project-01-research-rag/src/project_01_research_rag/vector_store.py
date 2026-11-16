from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "qasper_research"

def create_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(DATA_DIR),
        embedding_function=embeddings,
    )