
import time

import ir_measures
from ir_measures import Qrel, ScoredDoc, R, RR, nDCG

from dataset import load_scifact
from bm25_retriever import BM25Retriever
from semantic_retriever import SemanticRetriever
from hybrid_retriever import HybridRetriever
from embeddings import EmbeddingGenerator
from vector_index import VectorIndex


METRICS = [R@5, R@10, RR@10, nDCG@10]


def evaluate_retriever(
    name: str,
    retriever,
    queries,
    qrels,
) -> None:
    run = []
    total_latency = 0.0

    for query in queries:
        start = time.perf_counter()

        results = retriever.search(query["text"], top_k=10)

        total_latency += time.perf_counter() - start

        for rank, result in enumerate(results, start=1):
            run.append(
                ScoredDoc(
                    query_id=str(query["id"]),
                    doc_id=result.document_id,
                    score=float(11 - rank),
                )
            )

    scores = ir_measures.calc_aggregate(METRICS, qrels, run)

    print(f"\n{name} Evaluation")
    print("------------------------")

    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")

    print(
        f"Average latency: "
        f"{total_latency / len(queries) * 1000:.2f} ms"
    )


def main() -> None:
    corpus, queries, raw_qrels = load_scifact()

    qrels = [
        Qrel(
            query_id=str(qrel["query-id"]),
            doc_id=str(qrel["corpus-id"]),
            relevance=int(qrel["score"]),
        )
        for qrel in raw_qrels
    ]

    bm25 = BM25Retriever(corpus)

    generator = EmbeddingGenerator()
    semantic = SemanticRetriever(VectorIndex(generator))

    hybrid = HybridRetriever(
        bm25=bm25,
        semantic=semantic,
        candidate_k=50,
    )

    for name, retriever in [
        ("BM25", bm25),
        ("Semantic", semantic),
        ("Hybrid", hybrid),
    ]:
        evaluate_retriever(name, retriever, queries, qrels)


if __name__ == "__main__":
    main()
