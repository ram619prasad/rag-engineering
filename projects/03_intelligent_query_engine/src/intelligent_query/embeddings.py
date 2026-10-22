
import numpy as np
import tiktoken

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from models import Chunk


load_dotenv()

MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
PRICE_PER_MILLION_TOKENS = 0.02


class EmbeddingGenerator:
    def __init__(self, batch_size: int = 100) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.batch_size = batch_size
        self.model = MODEL
        self.dimensions = EMBEDDING_DIMENSIONS

        self.client = OpenAIEmbeddings(
            model=self.model,
            dimensions=self.dimensions,
            chunk_size=batch_size,
        )

        self.tokenizer = tiktoken.encoding_for_model(
            self.model
        )


    def estimate_cost(self, chunks: list[Chunk]) -> tuple[int, float]:
        total_tokens = sum(
            len(self.tokenizer.encode(chunk.text))
            for chunk in chunks
        )

        estimated_cost = (
            total_tokens / 1_000_000
        ) * PRICE_PER_MILLION_TOKENS

        return total_tokens, estimated_cost


    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        if not chunks:
            raise ValueError("chunks cannot be empty")

        texts = [chunk.text for chunk in chunks]

        vectors = self.client.embed_documents(texts)

        return np.asarray(vectors, dtype=np.float32)

    
    def embed_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("query cannot be empty")

        vector = self.client.embed_query(query)

        return np.asarray(vector, dtype=np.float32)
