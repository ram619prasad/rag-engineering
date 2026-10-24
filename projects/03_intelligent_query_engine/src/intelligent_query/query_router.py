
import time
from typing import Literal, TypedDict

from langgraph.graph import StateGraph, START, END

from dataset import load_scifact
from models import RetrievalResult
from query_classifier import QueryClassifier
from bm25_retriever import BM25Retriever
from embeddings import EmbeddingGenerator
from vector_index import VectorIndex
from semantic_retriever import SemanticRetriever
from hybrid_retriever import HybridRetriever


class RouterState(TypedDict):
    query: str
    strategy: Literal["bm25", "semantic", "hybrid"]
    results: list[RetrievalResult]


def build_router(
    classifier: QueryClassifier,
    bm25: BM25Retriever,
    semantic: SemanticRetriever,
    hybrid: HybridRetriever,
):
    # Step 1: Classify query.
    def classify(state: RouterState) -> dict:
        classification = classifier.classify(state["query"])

        print(
            f"Query type: {classification.query_type}"
        )
        print(
            f"Selected strategy: "
            f"{classification.retrieval_strategy}"
        )

        return {
            "strategy": classification.retrieval_strategy
        }

    # Step 2: Route according to classification.
    def choose_retriever(
        state: RouterState,
    ) -> Literal["bm25", "semantic", "hybrid"]:
        return state["strategy"]

    # Step 3: Retrieve using selected strategy.
    def search_bm25(state: RouterState) -> dict:
        return {
            "results": bm25.search(
                state["query"],
                top_k=5,
            )
        }

    def search_semantic(state: RouterState) -> dict:
        return {
            "results": semantic.search(
                state["query"],
                top_k=5,
            )
        }

    def search_hybrid(state: RouterState) -> dict:
        return {
            "results": hybrid.search(
                state["query"],
                top_k=5,
            )
        }

    # Step 4: Build LangGraph workflow.
    graph = StateGraph(RouterState)

    graph.add_node("classify", classify)
    graph.add_node("bm25", search_bm25)
    graph.add_node("semantic", search_semantic)
    graph.add_node("hybrid", search_hybrid)

    graph.add_edge(START, "classify")

    graph.add_conditional_edges(
        "classify",
        choose_retriever,
        {
            "bm25": "bm25",
            "semantic": "semantic",
            "hybrid": "hybrid",
        },
    )

    graph.add_edge("bm25", END)
    graph.add_edge("semantic", END)
    graph.add_edge("hybrid", END)

    return graph.compile()


def main() -> None:
    corpus, _, _ = load_scifact()

    bm25 = BM25Retriever(corpus)

    semantic = SemanticRetriever(
        VectorIndex(EmbeddingGenerator())
    )

    hybrid = HybridRetriever(
        bm25=bm25,
        semantic=semantic,
        candidate_k=50,
    )

    classifier = QueryClassifier()

    router = build_router(
        classifier=classifier,
        bm25=bm25,
        semantic=semantic,
        hybrid=hybrid,
    )

    queries = [
        # Short scientific claim
        "0-dimensional biomaterials show inductive properties.",

        # Scientific relationship
        "Inflammatory biomarkers can predict COPD exacerbations.",

        # Scientific comparison
        "How do different inflammatory biomarkers compare "
        "in predicting COPD exacerbations?",

        # Multiple information needs
        "How do inflammatory biomarkers predict COPD exacerbations "
        "and how do exacerbations affect patient survival?",
    ]

    for query in queries:
        start = time.perf_counter()

        output = router.invoke({
            "query": query,
        })

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        print(f"\nQuery: {query}")
        print(f"Route: {output['strategy']}")
        print(f"Latency: {latency_ms:.2f} ms")

        for rank, result in enumerate(
            output["results"],
            start=1,
        ):
            print(
                f"  {rank}. {result.title} "
                f"({result.document_id})"
            )


if __name__ == "__main__":
    main()
