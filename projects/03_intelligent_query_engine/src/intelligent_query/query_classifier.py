import time
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from models import QueryClassification


load_dotenv()


class QueryClassifier:
    def __init__(self) -> None:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )

        structured_llm = llm.with_structured_output(
            QueryClassification,
            method="json_schema",
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                Classify the user's query.

                Query types:
                - exact_match: Exact IDs, error codes, specific strings
                - factual: Requests for a specific fact
                - explanatory: Requests for concepts or explanations
                - comparison: Compares two or more concepts
                - troubleshooting: Diagnoses a problem
                - multi_hop: Requires connecting evidence across sources

                Suggest a retrieval strategy:
                - bm25: Exact terminology and identifiers
                - semantic: Conceptual or paraphrased questions
                - hybrid: Queries likely to benefit from both

                Return a brief reason.

                Do not answer the user's question.
                """
            ),
            ("human", "{query}"),
        ])

        self.chain = prompt | structured_llm

    
    def classify(self, query: str) -> QueryClassification:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        return self.chain.invoke({"query": query})


def main() -> None:
    classifier = QueryClassifier()

    queries = [
        "ERR_CONN_RESET_1042",
        "What is semantic search?",
        "Compare BM25 and vector search",
        "Why are my Kubernetes pods restarting?",
        "0-dimensional biomaterials show inductive properties.",
    ]

    for query in queries:
        start = time.perf_counter()

        result = classifier.classify(query)

        latency_ms = (time.perf_counter() - start) * 1000

        print(f"\nQuery: {query}")
        print(f"Type: {result.query_type}")
        print(f"Strategy: {result.retrieval_strategy}")
        print(f"Reason: {result.reason}")
        print(f"Classification latency: {latency_ms:.2f} ms")


if __name__ == "__main__":
    main()
