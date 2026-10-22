
import random
import time

import ir_measures
from ir_measures import Qrel, ScoredDoc, R

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

from dataset import load_scifact
from embeddings import EmbeddingGenerator
from vector_index import VectorIndex
from semantic_retriever import SemanticRetriever
from multi_query_retriever import SemanticRetrieverAdapter


# ---------------------------------------
# Configuration
# ---------------------------------------

SAMPLE_SIZE = 10
RANDOM_SEED = 42

BASELINE_K = 20
MULTI_QUERY_K = 5

# Original query + 3 generated queries
# Each retrieves up to 5 documents.
CANDIDATE_K = 20

METRICS = [R@CANDIDATE_K]


def add_to_run(
    run: list[ScoredDoc],
    query_id: str,
    document_ids: list[str],
) -> None:
    """Convert document IDs into ir-measures results."""

    for rank, doc_id in enumerate(document_ids, start=1):
        run.append(
            ScoredDoc(
                query_id=query_id,
                doc_id=doc_id,
                score=float(len(document_ids) + 1 - rank),
            )
        )


def evaluate(
    name: str,
    run: list[ScoredDoc],
    qrels: list[Qrel],
) -> None:
    """Calculate candidate recall using ir-measures."""

    scores = ir_measures.calc_aggregate(
        METRICS,
        qrels,
        run,
    )

    print(f"\n{name}")
    print("-" * 40)

    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")


def main() -> None:

    # ---------------------------------------
    # 1. Load SciFact
    # ---------------------------------------

    _, queries, raw_qrels = load_scifact()

    rng = random.Random(RANDOM_SEED)

    selected_queries = rng.sample(
        list(queries),
        k=min(SAMPLE_SIZE, len(queries)),
    )

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
    # 2. Initialize retrievers
    # ---------------------------------------

    semantic = SemanticRetriever(
        VectorIndex(EmbeddingGenerator())
    )

    adapter = SemanticRetrieverAdapter(
        semantic=semantic,
        top_k=MULTI_QUERY_K,
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    multi_query = MultiQueryRetriever.from_llm(
        retriever=adapter,
        llm=llm,
        include_original=True,
    )

    # ---------------------------------------
    # 3. Prepare evaluation runs
    # ---------------------------------------

    baseline_run = []
    multi_query_run = []

    baseline_time = 0.0
    multi_query_time = 0.0

    candidate_counts = []

    # ---------------------------------------
    # 4. Run retrieval
    # ---------------------------------------

    print(
        f"\nEvaluating {len(selected_queries)} "
        f"SciFact queries"
    )

    for index, query in enumerate(
        selected_queries,
        start=1,
    ):
        query_id = str(query["id"])
        query_text = query["text"]

        # Baseline semantic retrieval
        start = time.perf_counter()

        baseline_results = semantic.search(
            query_text,
            top_k=BASELINE_K,
        )

        baseline_time += time.perf_counter() - start

        baseline_ids = [
            result.document_id
            for result in baseline_results
        ]

        # Multi-query retrieval
        start = time.perf_counter()

        documents = multi_query.invoke(query_text)

        multi_query_time += time.perf_counter() - start

        # Deduplicated union returned by LangChain
        multi_ids = list(dict.fromkeys(
            str(doc.metadata["document_id"])
            for doc in documents
        ))

        candidate_counts.append(len(multi_ids))

        # Convert results into ir-measures format
        add_to_run(
            baseline_run,
            query_id,
            baseline_ids,
        )

        add_to_run(
            multi_query_run,
            query_id,
            multi_ids,
        )

        print(
            f"Query {index}/{len(selected_queries)} "
            f"| ID: {query_id} "
            f"| Multi-query candidates: {len(multi_ids)}"
        )

    # ---------------------------------------
    # 5. Evaluate using ir-measures
    # ---------------------------------------

    evaluate(
        name=f"Semantic Top-{BASELINE_K}",
        run=baseline_run,
        qrels=qrels,
    )

    evaluate(
        name="Multi-Query Candidate Pool",
        run=multi_query_run,
        qrels=qrels,
    )

    # ---------------------------------------
    # 6. Latency and candidate statistics
    # ---------------------------------------

    count = len(selected_queries)

    print("\nPerformance Comparison")
    print("-" * 40)

    print(f"Queries evaluated: {count}")

    print(
        f"Average multi-query candidates: "
        f"{sum(candidate_counts) / count:.2f}"
    )

    print(
        f"Average semantic latency: "
        f"{baseline_time / count * 1000:.2f} ms"
    )

    print(
        f"Average multi-query latency: "
        f"{multi_query_time / count * 1000:.2f} ms"
    )


if __name__ == "__main__":
    main()
