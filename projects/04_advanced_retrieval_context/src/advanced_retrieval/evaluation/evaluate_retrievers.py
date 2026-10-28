
import json

import ir_measures
from ir_measures import R, RR

from advanced_retrieval.retrievers.bm25_retriever import (
    BM25Retriever,
    CATALOG_PATH,
)
from advanced_retrieval.retrievers.semantic_retriever import (
    SemanticRetriever,
)
from advanced_retrieval.retrievers.hybrid_retriever import (
    create_hybrid_retriever,
)


BENCHMARK_PATH = (
    CATALOG_PATH.parent / "benchmark_dev_generated.json"
)

METRICS = [
    R @ 5,
    R @ 10,
    RR @ 10,
]


def evaluate(
    retriever,
    questions: list[dict],
) -> dict:
    qrels = []
    run = []

    for question in questions:
        question_id = question["question_id"]

        # Ground-truth relevant sections
        for section_id in question["relevant_section_ids"]:
            qrels.append(
                ir_measures.Qrel(
                    query_id=question_id,
                    doc_id=section_id,
                    relevance=1,
                )
            )

        # Retrieve top 10 sections
        results = retriever.invoke(question["question"])[:10]

        for rank, document in enumerate(results, start=1):
            run.append(
                ir_measures.ScoredDoc(
                    query_id=question_id,
                    doc_id=document.metadata["section_id"],
                    score=float(11 - rank),
                )
            )

    return ir_measures.calc_aggregate(
        METRICS,
        qrels,
        run,
    )
    


def analyze_retrieval_failure(
    questions: list[dict],
    semantic: SemanticRetriever,
    hybrid,
):
    for question in questions:
        query = question["question"]
        relevant = set(question["relevant_section_ids"])

        semantic_results = semantic.invoke(query)[:10]
        hybrid_results = hybrid.invoke(query)[:10]

        semantic_ids = [
            doc.metadata["section_id"]
            for doc in semantic_results
        ]

        hybrid_ids = [
            doc.metadata["section_id"]
            for doc in hybrid_results
        ]

        semantic_hits = len(relevant.intersection(semantic_ids))
        hybrid_hits = len(relevant.intersection(hybrid_ids))

        if semantic_hits <= hybrid_hits:
            continue

        print("\n=== RETRIEVAL FAILURE ANALYSIS ===")
        print(f"Question: {query}")
        print(f"Relevant sections: {sorted(relevant)}")

        for name, ids in [
            ("Semantic", semantic_ids),
            ("Hybrid", hybrid_ids),
        ]:
            print(f"\n{name} results:")

            for rank, section_id in enumerate(ids, start=1):
                marker = "✓" if section_id in relevant else " "
                print(f"{rank:2}. [{marker}] {section_id}")

        return

    print("\nNo case found where Semantic outperformed Hybrid at top 10.")



def main():
    sections = json.loads(
        CATALOG_PATH.read_text(encoding="utf-8")
    )

    questions = json.loads(
        BENCHMARK_PATH.read_text(encoding="utf-8")
    )["questions"]

    # Both base retrievers return 20 candidates.
    bm25 = BM25Retriever(
        sections=sections,
        k=20,
    )

    semantic = SemanticRetriever(k=20)

    if not semantic.has_index():
        semantic.index(sections)

    hybrid = create_hybrid_retriever(
        bm25=bm25,
        semantic=semantic,
    )

    retrievers = {
        "BM25": bm25,
        "Semantic": semantic,
        "Hybrid": hybrid,
    }

    print(f"\nEvaluating {len(questions)} questions...\n")

    results = {}

    for name, retriever in retrievers.items():
        print(f"Evaluating {name}...")

        results[name] = evaluate(
            retriever,
            questions,
        )

    print("\n=== RETRIEVAL EVALUATION ===")
    print(
        f"{'Retriever':<12}"
        f"{'Recall@5':>12}"
        f"{'Recall@10':>12}"
        f"{'MRR@10':>12}"
    )

    for name, metrics in results.items():
        print(
            f"{name:<12}"
            f"{metrics[R @ 5]:>12.4f}"
            f"{metrics[R @ 10]:>12.4f}"
            f"{metrics[RR @ 10]:>12.4f}"
        )
        
        
    analyze_retrieval_failure(
        questions=questions,
        semantic=semantic,
        hybrid=hybrid,
    )


if __name__ == "__main__":
    main()
