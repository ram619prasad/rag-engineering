
from pathlib import Path
from datasets import DatasetDict, load_dataset, load_from_disk
from collections import Counter


DATASET_NAME = "Hyukkyu/beir-scifact"
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_DIR / "datasets" / "scifact"

def load_scifact():
    # Reuse the local dataset if it already exists
    if DATASET_DIR.exists():
        print("Loading SciFact from local storage...")
        dataset = load_from_disk(str(DATASET_DIR))
    else:
        print("Downloading SciFact from Hugging Face...")

        dataset = DatasetDict({
            "corpus": load_dataset(
                DATASET_NAME, "corpus", split="test"
            ),
            "queries": load_dataset(
                DATASET_NAME, "queries", split="test"
            ),
            "qrels": load_dataset(
                DATASET_NAME, "qrels", split="test"
            ),
        })

        # Store locally for future runs
        dataset.save_to_disk(str(DATASET_DIR))
        print("SciFact saved locally.")

    return (
        dataset["corpus"],
        dataset["queries"],
        dataset["qrels"],
    )


def validate_scifact(corpus, queries, qrels) -> None:
    # Collect all IDs
    document_ids = [str(doc["id"]) for doc in corpus]
    query_ids = [str(query["id"]) for query in queries]
    document_id_set = set(document_ids)
    query_id_set = set(query_ids)

    # 1. Ensure IDs are unique
    if len(document_ids) != len(document_id_set):
        raise ValueError("Duplicate document IDs found")

    if len(query_ids) != len(query_id_set):
        raise ValueError("Duplicate query IDs found")

    # 2. Validate qrel references
    missing_documents = set()
    missing_queries = set()
    relevant_counts = Counter()

    for qrel in qrels:
        query_id = str(qrel["query-id"])
        document_id = str(qrel["corpus-id"])
        score = int(qrel["score"])

        if query_id not in query_id_set:
            missing_queries.add(query_id)

        if document_id not in document_id_set:
            missing_documents.add(document_id)

        if score > 0:
            relevant_counts[query_id] += 1

    if missing_queries:
        raise ValueError(
            f"Qrels reference missing queries: {sorted(missing_queries)[:5]}"
        )

    if missing_documents:
        raise ValueError(
            f"Qrels reference missing documents: {sorted(missing_documents)[:5]}"
        )

    # 3. Report dataset characteristics
    print("\nDataset Validation")
    print("------------------")
    print(f"Unique documents: {len(document_id_set)}")
    print(f"Unique queries: {len(query_id_set)}")
    print(f"Relevance judgments: {len(qrels)}")
    print(f"Queries with relevant documents: {len(relevant_counts)}")

    if relevant_counts:
        print(
            "Average relevant documents per judged query: "
            f"{sum(relevant_counts.values()) / len(relevant_counts):.2f}"
        )

    print("Validation: PASSED")


def main() -> None:
    corpus, queries, qrels = load_scifact()

    print(f"Documents: {len(corpus)}")
    print(f"Queries: {len(queries)}")
    print(f"Qrels: {len(qrels)}")

    validate_scifact(corpus, queries, qrels)
    

if __name__ == "__main__":
    main()