from datasets import load_dataset

DATASET_NAME = "Hyukkyu/beir-scifact"

def load_scifact():
    corpus = load_dataset(
        DATASET_NAME, "corpus", split="test"
    )
    
    queries = load_dataset(
        DATASET_NAME, "corpus", split="test"
    )
    
    qrels = load_dataset(
        DATASET_NAME, "qrels", split="test"
    )

    return corpus, queries, qrels

def main() -> None:
    corpus, queries, qrels = load_scifact()

    print(f"Documents: {len(corpus)}")  # 5183
    print(f"Test queries: {len(queries)}")  # 300
    print(f"Test qrels: {len(qrels)}")  # 339

    print("\nSample document:")
    print(corpus[0])

    print("\nSample query:")
    print(queries[0])

    print("\nSample relevance judgment:")
    print(qrels[0])


if __name__ == "__main__":
    main()