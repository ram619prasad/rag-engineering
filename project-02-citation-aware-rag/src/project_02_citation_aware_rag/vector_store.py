from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "chroma"
COLLECTION_NAME = "qasper_citation_aware"



def create_vector_store() -> Chroma:
    return Chroma(
        collection_name=str(COLLECTION_NAME),
        persist_directory=str(DATA_DIR),
        embedding_function=OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
    )