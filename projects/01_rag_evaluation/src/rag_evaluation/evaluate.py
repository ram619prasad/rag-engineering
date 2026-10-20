
from load_dataset import load_dataset
from metrics import hit_rate_at_k, precision_at_k, recall_at_k


# Configuration A: existing baseline
simulated_results = {
    "q001": ["refund_01", "shipping_01", "warranty_01"],
    "q002": ["shipping_01", "warranty_01"],
    "q003": ["shipping_01", "refund_01"],
    "q004": ["refund_01", "shipping_01", "warranty_01"],
    "q005": ["shipping_01", "warranty_01"],
}

# Configuration B: alternative retrieval results
alternative_results = {
    "q001": ["refund_01", "warranty_01", "shipping_01"],
    "q002": ["warranty_01", "shipping_01"],
    "q003": ["refund_01", "warranty_01"],
    "q004": ["refund_01", "shipping_01", "warranty_01"],
    "q005": ["refund_01", "shipping_01"],
}


def evaluate_configuration(
    records,
    retrieved_results: dict[str, list[str]],
    k: int = 3,
) -> dict[str, float]:
    results = []

    for record in records:
        retrieved = retrieved_results[record.id]
        relevant = record.relevant_chunk_ids

        results.append({
            "hit_rate": hit_rate_at_k(retrieved, relevant, k),
            "precision": precision_at_k(retrieved, relevant, k),
            "recall": recall_at_k(retrieved, relevant, k),
        })

    if not results:
        raise ValueError("Cannot evaluate an empty dataset.")

    count = len(results)

    return {
        "hit_rate": sum(r["hit_rate"] for r in results) / count,
        "precision": sum(r["precision"] for r in results) / count,
        "recall": sum(r["recall"] for r in results) / count,
    }


def main() -> None:
    records = load_dataset()

    baseline = evaluate_configuration(records, simulated_results, k=3)
    alternative = evaluate_configuration(records, alternative_results, k=3)

    print("RAG Retrieval Evaluation — K=3")
    print("-" * 45)

    print(f"{'Metric':<15}{'Baseline':>12}{'Alternative':>15}")

    for metric in ("hit_rate", "precision", "recall"):
        print(
            f"{metric:<15}"
            f"{baseline[metric]:>12.2f}"
            f"{alternative[metric]:>15.2f}"
        )


if __name__ == "__main__":
    main()
