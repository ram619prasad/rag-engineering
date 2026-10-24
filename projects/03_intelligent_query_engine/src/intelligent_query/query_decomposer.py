
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from models import DecomposedQuery
from embeddings import EmbeddingGenerator
from vector_index import VectorIndex
from semantic_retriever import SemanticRetriever

load_dotenv()


class QueryDecomposer:
    def __init__(self):
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You decompose complex questions for document retrieval.

                Rules:
                - Break the question into 2-4 focused subquestions
                  only when multiple information needs exist.
                - Preserve the original intent and entities.
                - Each subquestion must have a distinct purpose.
                - Do not generate paraphrases of the same question.
                - Do not invent assumptions or facts.
                - If the question is already simple, return it
                  as the only subquestion.
                - Do not answer any question.
                """,
            ),
            ("human", "{query}"),
        ])

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )

        self.chain = prompt | llm.with_structured_output(
            DecomposedQuery,
            method="json_schema",
        )

    def decompose(self, query: str) -> DecomposedQuery:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        return self.chain.invoke({"query": query})



def retrieve_subquestion_evidence(
    query: str,
    decomposer: QueryDecomposer,
    retriever: SemanticRetriever,
    top_k: int = 3,
) -> None:
    decomposition = decomposer.decompose(query)
    evidence_by_question = {}

    print(f"\nOriginal Question: {query}")
    print(f"Subquestions: {len(decomposition.subquestions)}")

    all_document_ids = set()

    for index, subquestion in enumerate(
        decomposition.subquestions,
        start=1,
    ):
        print(f"\nSubquestion {index}: {subquestion}")

        results = retriever.search(
            subquestion,
            top_k=top_k,
        )
        
        evidence_by_question[subquestion] = {
            result.document_id for result in results
        }

        for rank, result in enumerate(results, start=1):
            print(
                f"  {rank}. {result.title} "
                f"(ID: {result.document_id}, "
                f"Score: {result.score:.4f})"
            )

            all_document_ids.add(result.document_id)

    for question, document_ids in evidence_by_question.items():
        print(f"\n{question}")
        print(f"Retrieved document IDs: {document_ids}")

    all_ids = [
        doc_id
        for document_ids in evidence_by_question.values()
        for doc_id in document_ids
    ]

    print(f"\nTotal retrieved: {len(all_ids)}")
    print(f"Unique documents: {len(set(all_ids))}")
    print(f"Duplicate occurrences: {len(all_ids) - len(set(all_ids))}")




def main() -> None:
    decomposer = QueryDecomposer()

    semantic = SemanticRetriever(
        VectorIndex(EmbeddingGenerator())
    )

    query = (
        "How do inflammatory biomarkers predict COPD "
        "exacerbations, and how do exacerbations "
        "affect patient survival?"
    )

    retrieve_subquestion_evidence(
        query=query,
        decomposer=decomposer,
        retriever=semantic,
        top_k=3,
    )


if __name__ == "__main__":
    main()
