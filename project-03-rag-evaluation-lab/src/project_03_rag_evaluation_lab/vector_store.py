from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()


PARENT_DIR = Path(__file__).resolve().parents[2]
CROMA_DIR = PARENT_DIR / "data" / "croma"
COLLECTION_NAME = "qasper"


def create_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CROMA_DIR),
        embedding_function=OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
    )