def hit_rate_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")

    # 1. Consider only the first k retrieved chunks.
    top_k_retrieved = retrieved_chunk_ids[:k]
    
    # Convert to set for O(1) lookups
    relevant_set = set(relevant_chunk_ids)
    
    # 2. Return 1.0 if any is relevant; otherwise return 0.0.
    if any(chunk_id in relevant_set for chunk_id in top_k_retrieved):
        return 1.0
        
    return 0.0


def precision_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int,
) -> float:
    # 1. Validate k.
    if k <= 0:
        raise ValueError("k must be a positive integer.")
        
    # 2. Keep only the top k retrieved chunks.
    top_k_retrieved = retrieved_chunk_ids[:k]
    
    # 5. Return 0.0 if no chunks were retrieved (done early to prevent division by zero).
    if not top_k_retrieved:
        return 0.0
        
    # Convert relevant IDs to a set for O(1) lookup time
    relevant_set = set(relevant_chunk_ids)
    
    # 3. Count how many retrieved chunks are relevant.
    relevant_count = sum(1 for chunk_id in top_k_retrieved if chunk_id in relevant_set)
    
    # 4. Divide by the number of chunks actually retrieved.
    return relevant_count / len(top_k_retrieved)


def recall_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int,
) -> float:
    # 1. Validate k
    if k <= 0:
        raise ValueError("k must be a positive integer.")
        
    # 2. Prevent division by zero if there are no relevant chunks in the ground truth
    if not relevant_chunk_ids:
        return 0.0
        
    # 3. Keep only the top k retrieved chunks
    top_k_retrieved = retrieved_chunk_ids[:k]
    
    # Convert to set for O(1) lookups
    relevant_set = set(relevant_chunk_ids)
    
    # 4. Count how many of the top K retrieved chunks are relevant
    relevant_count = sum(1 for chunk_id in top_k_retrieved if chunk_id in relevant_set)
    
    # 5. Divide by the TOTAL number of relevant chunks (not the retrieved count)
    return relevant_count / len(relevant_chunk_ids)


def test_hit_rate():
    relevant = ["refund_01", "refund_02"]

    # Case 1: Relevant chunk appears in top 3
    retrieved = ["shipping_01", "warranty_01", "refund_02"]
    print(hit_rate_at_k(retrieved, relevant, 3))

    print(hit_rate_at_k(retrieved, relevant, 2))

    retrieved = ["shipping_01", "warranty_01"]
    print(hit_rate_at_k(retrieved, relevant, 2))


def test_presision():
    relevant = ["refund_01", "refund_02"]

    # Case 1: Two relevant chunks among three retrieved
    retrieved = ["refund_01", "shipping_01", "refund_02"]
    print(precision_at_k(retrieved, relevant, 3))  # Expected: 0.6667

    # Case 2: One relevant chunk among the top two
    retrieved = ["shipping_01", "refund_01", "refund_02"]
    print(precision_at_k(retrieved, relevant, 2))  # Expected: 0.5

    # Case 3: No chunks retrieved
    print(precision_at_k([], relevant, 3))  # Expected: 0.0


def test_recall():
    print(recall_at_k(["a", "c"], ["a", "b"], 2))       # 0.5
    print(recall_at_k(["c", "d"], ["a", "b"], 2))       # 0.0
    print(recall_at_k(["c", "a", "b"], ["a", "b"], 1))  # 0.0
    print(recall_at_k(["a", "c"], [], 2))               # 0.0


if __name__ == "__main__":
    # test_hit_rate()
    # test_presision()
    test_recall()