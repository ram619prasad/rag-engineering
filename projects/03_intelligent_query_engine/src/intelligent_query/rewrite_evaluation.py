
import time

import ir_measures
from ir_measures import Qrel, ScoredDoc, R, RR, nDCG

from dataset import load_scifact
from embeddings import EmbeddingGenerator
from vector_index import VectorIndex
from semantic_retriever import SemanticRetriever
from query_rewriter import QueryRewriter


METRICS = [R@5, R@10, RR@10, nDCG@10]

TOP_K = 10

# Set to 5 for testing, None for all 300 queries.
MAX_QUERIES = None

# Maximum concurrent LLM rewriting requests.
MAX_CONCURRENCY = 20


def add_to_run(
    run: list,
    query_id: str,
    results: list,
) -> None:
    """Convert retrieval results into ir-measures format."""

    for rank, result in enumerate(results[:TOP_K], start=1):
        run.append(
            ScoredDoc(
                query_id=str(query_id),
                doc_id=result.document_id,
                score=float(TOP_K + 1 - rank),
            )
        )


def evaluate(
    name: str,
    run: list,
    qrels: list,
) -> None:
    """Evaluate a retrieval run using ir-measures."""

    scores = ir_measures.calc_aggregate(
        METRICS,
        qrels,
        run,
    )

    print(f"\n{name} Semantic Evaluation")
    print("-" * 35)

    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")


def main() -> None:

    # ---------------------------------------
    # 1. Load SciFact
    # ---------------------------------------

    corpus, queries, raw_qrels = load_scifact()

    selected_queries = (
        queries.select(
            range(min(MAX_QUERIES, len(queries)))
        )
        if MAX_QUERIES is not None
        else queries
    )

    query_count = len(selected_queries)

    if query_count == 0:
        raise ValueError("No queries selected for evaluation.")

    selected_ids = {
        str(query["id"])
        for query in selected_queries
    }

    qrels = [
        Qrel(
            query_id=str(row["query-id"]),
            doc_id=str(row["corpus-id"]),
            relevance=int(row["score"]),
        )
        for row in raw_qrels
        if str(row["query-id"]) in selected_ids
    ]

    # ---------------------------------------
    # 2. Initialize existing components
    # ---------------------------------------

    semantic = SemanticRetriever(
        VectorIndex(EmbeddingGenerator())
    )

    rewriter = QueryRewriter()

    original_run = []
    rewritten_run = []

    original_search_time = 0.0
    rewritten_search_time = 0.0

    # ---------------------------------------
    # 3. Batch rewrite queries
    # ---------------------------------------

    query_texts = [
        query["text"]
        for query in selected_queries
    ]

    print(f"\nEvaluating {query_count} SciFact queries")
    print(
        f"Batch rewriting with max concurrency: "
        f"{MAX_CONCURRENCY}"
    )

    start = time.perf_counter()

    rewritten_queries = rewriter.rewrite_batch(
        query_texts,
        max_concurrency=MAX_CONCURRENCY,
    )

    rewrite_time = time.perf_counter() - start

    if len(rewritten_queries) != query_count:
        raise ValueError(
            "Number of rewritten queries does not "
            "match the number of original queries."
        )

    print(
        f"Batch rewriting completed in "
        f"{rewrite_time:.2f} seconds"
    )

    # ---------------------------------------
    # 4. Evaluate original and rewritten
    # ---------------------------------------

    for index, (query, rewritten_text) in enumerate(
        zip(selected_queries, rewritten_queries),
        start=1,
    ):
        query_id = str(query["id"])
        original_text = query["text"]

        # Original semantic retrieval
        start = time.perf_counter()

        original_results = semantic.search(
            original_text,
            top_k=TOP_K,
        )

        original_search_time += (
            time.perf_counter() - start
        )

        # Rewritten semantic retrieval
        start = time.perf_counter()

        rewritten_results = semantic.search(
            rewritten_text,
            top_k=TOP_K,
        )

        rewritten_search_time += (
            time.perf_counter() - start
        )

        add_to_run(
            original_run,
            query_id,
            original_results,
        )

        add_to_run(
            rewritten_run,
            query_id,
            rewritten_results,
        )

        if index % 25 == 0 or index == query_count:
            print(
                f"Retrieval evaluated: "
                f"{index}/{query_count}"
            )

    # ---------------------------------------
    # 5. Calculate retrieval metrics
    # ---------------------------------------

    evaluate(
        name="Original",
        run=original_run,
        qrels=qrels,
    )

    evaluate(
        name="Rewritten",
        run=rewritten_run,
        qrels=qrels,
    )

    # ---------------------------------------
    # 6. Latency report
    # ---------------------------------------

    original_avg = (
        original_search_time / query_count * 1000
    )

    rewritten_avg = (
        rewritten_search_time / query_count * 1000
    )

    rewrite_avg = (
        rewrite_time / query_count * 1000
    )

    print("\nLatency Report")
    print("-" * 35)

    print(f"Queries evaluated: {query_count}")

    print(
        f"Total batch rewriting time: "
        f"{rewrite_time:.2f} seconds"
    )

    print(
        f"Amortized rewriting time/query: "
        f"{rewrite_avg:.2f} ms"
    )

    print(
        f"Original semantic retrieval/query: "
        f"{original_avg:.2f} ms"
    )

    print(
        f"Rewritten semantic retrieval/query: "
        f"{rewritten_avg:.2f} ms"
    )

    print(
        f"Amortized rewrite + retrieval/query: "
        f"{rewrite_avg + rewritten_avg:.2f} ms"
    )


if __name__ == "__main__":
    main()
