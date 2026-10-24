
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from embeddings import EmbeddingGenerator
from vector_index import VectorIndex
from semantic_retriever import SemanticRetriever


class HyDERetriever:
    def __init__(self, semantic: SemanticRetriever):
        self.semantic = semantic

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You generate hypothetical scientific passages
                for document retrieval.

                Given a scientific query or claim:
                - Write a short passage resembling a relevant
                  scientific abstract.
                - Preserve the main subject and terminology.
                - Include potentially relevant scientific concepts.
                - Do not reference imaginary studies or citations.
                - Do not claim the passage is verified evidence.
                - Return only the hypothetical passage.

                The passage is used for retrieval, not as
                a factual answer.
                """
            ),
            ("human", "{query}"),
        ])

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )

        self.chain = prompt | llm | StrOutputParser()

    def search(self, query: str, top_k: int = 5):
        if not query.strip():
            raise ValueError("Query cannot be empty")

        # Step 1: Generate hypothetical document.
        hypothetical_document = self.chain.invoke(
            {"query": query}
        ).strip()

        if not hypothetical_document:
            raise ValueError("LLM returned an empty hypothetical document")

        # Step 2: Embed hypothetical text and retrieve real documents.
        # SemanticRetriever.search() already handles both.
        results = self.semantic.search(
            hypothetical_document,
            top_k=top_k,
        )

        return hypothetical_document, results


def main() -> None:
    semantic = SemanticRetriever(
        VectorIndex(EmbeddingGenerator())
    )

    hyde = HyDERetriever(semantic)

    query = (
        "0-dimensional biomaterials show inductive properties."
    )

    # Baseline semantic retrieval
    baseline_results = semantic.search(
        query,
        top_k=5,
    )

    # HyDE retrieval
    hypothetical_document, hyde_results = hyde.search(
        query,
        top_k=5,
    )

    print(f"\nOriginal Query: {query}")

    print("\nGenerated Hypothetical Document:")
    print(hypothetical_document)

    print("\nSemantic Top 5:")
    for rank, result in enumerate(baseline_results, 1):
        print(
            f"{rank}. {result.title} "
            f"(ID: {result.document_id})"
        )

    print("\nHyDE Top 5:")
    for rank, result in enumerate(hyde_results, 1):
        print(
            f"{rank}. {result.title} "
            f"(ID: {result.document_id})"
        )


if __name__ == "__main__":
    main()
