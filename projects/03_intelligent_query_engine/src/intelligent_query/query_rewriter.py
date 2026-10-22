
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


load_dotenv()


class QueryRewriter:
    def __init__(self) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a search query rewriting assistant.

                    Rewrite the user's query to improve retrieval
                    from a technical or scientific document collection.

                    Rules:
                    - Preserve the original intent.
                    - Use clear and searchable terminology.
                    - Preserve exact identifiers, numbers and entities.
                    - Do not invent facts, assumptions or constraints.
                    - Do not answer the query.
                    - Return only one rewritten query.
                    - If already clear, return the original query.
                    """,
                ),
                ("human", "{query}"),
            ]
        )

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )

        self.chain = prompt | llm | StrOutputParser()

    def rewrite(self, query: str) -> str:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        rewritten_query = self.chain.invoke(
            {"query": query}
        ).strip()

        return rewritten_query or query


    def rewrite_batch(
        self,
        queries: list[str],
        max_concurrency: int = 5,
    ) -> list[str]:

        if not queries:
            return []

        if any(not query.strip() for query in queries):
            raise ValueError("Queries cannot be empty")

        rewritten_queries = self.chain.batch(
            [{"query": query} for query in queries],
            config={"max_concurrency": max_concurrency},
        )

        return [
            rewritten.strip() or original
            for original, rewritten in zip(
                queries, rewritten_queries
            )
        ]


def main() -> None:
    rewriter = QueryRewriter()

    queries = [
        "Why are my Kubernetes pods dying?",
        "How does vector db work?",
        "Compare BM25 and semantic search",
        "ERR_CONN_RESET_1042",
        "0-dimensional biomaterials show inductive properties.",
    ]

    for query in queries:
        rewritten = rewriter.rewrite(query)

        print(f"\nOriginal:  {query}")
        print(f"Rewritten: {rewritten}")


if __name__ == "__main__":
    main()
