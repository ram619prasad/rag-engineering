import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from models import Chunk

load_dotenv()


class EmbeddingGenerator:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
    ) -> None:
        self.client = OpenAIEmbeddings(model=model)

    
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("texts cannot be empty.")

        if any(not text.strip() for text in texts):
            raise ValueError("texts cannot contain empty strings.")

        vectors = self.client.embed_documents(texts)
        return np.asarray(vectors, dtype=np.float32)

    
    def embed_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("query cannot be empty.")

        vector = self.client.embed_query(query)
        return np.asarray(vector, dtype=np.float32)


    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        return self.embed_texts([chunk.content for chunk in chunks])


def main() -> None:
    generator = EmbeddingGenerator()

    sample_texts = [
        "How can I spend less on my cluster?",
        "Reducing Kubernetes costs by right-sizing resources.",
        "How do I reset a database connection?",
    ]

    vectors = generator.embed_texts(sample_texts)

    print(f"Texts embedded: {len(sample_texts)}")
    print(f"Embedding matrix shape: {vectors.shape}")
    print(f"Vector dimension: {vectors.shape[1]}")

    query_vector = generator.embed_query(
        "How can I reduce infrastructure spending?"
    )
    print(f"Query vector shape: {query_vector.shape}")


if __name__ == "__main__":
    main()
