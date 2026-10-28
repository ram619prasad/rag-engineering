
import json

import ir_measures
from ir_measures import R, RR, Qrel, ScoredDoc, calc_aggregate

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
from advanced_retrieval.retrievers.contextual_retriever import (
    load_contextual_sections,
    CHROMA_DIR,
)
from advanced_retrieval.retrievers.parent_child_retriever import (
    create_parent_child_retriever,
)


BENCHMARK_PATH = (
    CATALOG_PATH.parent / "benchmark_dev_generated.json"
)

METRICS = [R @ 5, R @ 10, RR @ 10]


def evaluate(retriever, questions: list[dict]) -> dict:
    qrels = []
    run = []

    for question in questions:
        question_id = question["question_id"]

        for section_id in question["relevant_section_ids"]:
            qrels.append(
                Qrel(
                    query_id=question_id,
                    doc_id=section_id,
                    relevance=1,
                )
            )

        documents = retriever.invoke(question["question"])[:10]

        for rank, document in enumerate(documents, start=1):
            run.append(
                ScoredDoc(
                    query_id=question_id,
                    doc_id=document.metadata["section_id"],
                    score=float(11 - rank),
                )
            )

    return calc_aggregate(
        METRICS,
        qrels,
        run,
    )


def _evaluate_baseline(
    sections: list[dict],
    questions: list[dict],
) -> dict:
    print("\n=== BASELINE RETRIEVAL ===")

    bm25 = BM25Retriever(
        sections=sections,
        k=20,
    )

    semantic = SemanticRetriever(k=20)

    if not semantic.has_index():
        print("Indexing baseline sections...")
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

    results = {}

    for name, retriever in retrievers.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate(retriever, questions)

    return results


def _evaluate_contextual(
    questions: list[dict],
) -> dict:
    print("\n=== CONTEXTUAL RETRIEVAL ===")

    sections = load_contextual_sections()

    bm25 = BM25Retriever(
        sections=sections,
        k=20,
    )

    semantic = SemanticRetriever(
        k=20,
        collection_name="engineering_contextual_baseline",
        persist_directory=CHROMA_DIR,
    )

    # Contextual indexing is handled by contextual_retrieval.py.
    if not semantic.has_index():
        raise RuntimeError(
            "Contextual index not found. "
            "Run contextual_retrieval.py first."
        )

    hybrid = create_hybrid_retriever(
        bm25=bm25,
        semantic=semantic,
    )

    retrievers = {
        "Contextual BM25": bm25,
        "Contextual Semantic": semantic,
        "Contextual Hybrid": hybrid,
    }

    results = {}

    for name, retriever in retrievers.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate(retriever, questions)

    return results


def _print_results(results: dict) -> None:
    print("\n=== BASELINE VS CONTEXTUAL RETRIEVAL ===")

    print(
        f"{'Retriever':<24}"
        f"{'Recall@5':>12}"
        f"{'Recall@10':>12}"
        f"{'MRR@10':>12}"
    )

    for name, metrics in results.items():
        print(
            f"{name:<24}"
            f"{metrics[R @ 5]:>12.4f}"
            f"{metrics[R @ 10]:>12.4f}"
            f"{metrics[RR @ 10]:>12.4f}"
        )
        

def _evaluate_parent_child(
    sections: list[dict],
    questions: list[dict],
) -> dict:
    """Evaluate parent-child retrieval using parent section IDs."""

    print("\n=== PARENT-CHILD RETRIEVAL ===")

    retriever = create_parent_child_retriever(
        sections=sections,
        child_chunk_size=400,
        child_chunk_overlap=60,
        search_k=40,
    )

    print("Evaluating Parent-Child...")

    return {
        "Parent-Child": evaluate(
            retriever,
            questions,
        )
    }



def main():
    sections = json.loads(
        CATALOG_PATH.read_text(encoding="utf-8")
    )

    questions = json.loads(
        BENCHMARK_PATH.read_text(encoding="utf-8")
    )["questions"]

    print(f"Evaluating {len(questions)} development questions...")

    baseline_results = _evaluate_baseline(
        sections=sections,
        questions=questions,
    )

    contextual_results = _evaluate_contextual(
        questions=questions,
    )

    parent_child_results = _evaluate_parent_child(
        sections=sections,
        questions=questions,
    )

    combined_results = {
        **baseline_results,
        **contextual_results,
        **parent_child_results,
    }

    _print_results(combined_results)


if __name__ == "__main__":
    main()
