
import random
import time

import ir_measures
from ir_measures import Qrel, ScoredDoc, R, RR, nDCG

from dataset import load_scifact
from embeddings import EmbeddingGenerator
from vector_index import VectorIndex
from semantic_retriever import SemanticRetriever
from hyde_retriever import HyDERetriever


SAMPLE_SIZE = 10
RANDOM_SEED = 42
TOP_K = 10

METRICS = [R@5, R@10, RR@10, nDCG@10]


def add_to_run(run, query_id, results):
    for rank, result in enumerate(results, start=1):
        run.append(
            ScoredDoc(
                query_id=query_id,
                doc_id=result.document_id,
                score=float(TOP_K + 1 - rank),
            )
        )


def main():
    _, queries, raw_qrels = load_scifact()

    rng = random.Random(RANDOM_SEED)
    selected = rng.sample(
        list(queries),
        k=min(SAMPLE_SIZE, len(queries)),
    )

    selected_ids = {str(q["id"]) for q in selected}

    qrels = [
        Qrel(
            query_id=str(row["query-id"]),
            doc_id=str(row["corpus-id"]),
            relevance=int(row["score"]),
        )
        for row in raw_qrels
        if str(row["query-id"]) in selected_ids
    ]

    semantic = SemanticRetriever(
        VectorIndex(EmbeddingGenerator())
    )
    hyde = HyDERetriever(semantic)

    baseline_run = []
    hyde_run = []
    baseline_time = 0.0
    hyde_time = 0.0

    for index, query in enumerate(selected, start=1):
        query_id = str(query["id"])
        query_text = query["text"]

        start = time.perf_counter()
        baseline_results = semantic.search(
            query_text, top_k=TOP_K
        )
        baseline_time += time.perf_counter() - start

        start = time.perf_counter()
        _, hyde_results = hyde.search(
            query_text, top_k=TOP_K
        )
        hyde_time += time.perf_counter() - start

        add_to_run(baseline_run, query_id, baseline_results)
        add_to_run(hyde_run, query_id, hyde_results)

        print(f"Evaluated {index}/{len(selected)} queries")

    for name, run in [
        ("Semantic Baseline", baseline_run),
        ("HyDE", hyde_run),
    ]:
        scores = ir_measures.calc_aggregate(
            METRICS, qrels, run
        )

        print(f"\n{name}")
        print("-" * 30)

        for metric, value in scores.items():
            print(f"{metric}: {value:.4f}")

    count = len(selected)

    print("\nAverage Latency")
    print("-" * 30)
    print(
        f"Semantic: "
        f"{baseline_time / count * 1000:.2f} ms"
    )
    print(
        f"HyDE (generation + retrieval): "
        f"{hyde_time / count * 1000:.2f} ms"
    )


if __name__ == "__main__":
    main()
