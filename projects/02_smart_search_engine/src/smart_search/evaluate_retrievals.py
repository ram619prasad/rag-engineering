
import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from bm25_search import BM25Search
from chunking import chunk_documents
from load_documents import load_documents


async def main() -> None:
    load_dotenv()

    # 1. Load and index documents
    documents = load_documents()
    chunks = chunk_documents(documents)
    retriever = BM25Search(chunks)

    # 2. Evaluation question and expected answer
    query = "How can I reduce Kubernetes infrastructure costs?"

    reference = (
        "Reduce Kubernetes costs by right-sizing CPU and memory "
        "requests, removing idle workloads, improving bin packing, "
        "and using autoscaling while monitoring latency and reliability."
    )

    # 3. Retrieve actual passages
    results = retriever.search(query, top_k=3)

    retrieved_contexts = [
        result["content"] for result in results
    ]

    # 4. Build Ragas sample
    sample = SingleTurnSample(
        user_input=query,
        retrieved_contexts=retrieved_contexts,
        reference=reference,
    )

    # 5. Configure evaluation LLM
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o-mini", temperature=0)
    )

    # 6. Initialize Ragas metrics
    precision_metric = LLMContextPrecisionWithReference(
        llm=evaluator_llm
    )

    recall_metric = LLMContextRecall(
        llm=evaluator_llm
    )

    # 7. Evaluate
    precision = await precision_metric.single_turn_ascore(sample)
    recall = await recall_metric.single_turn_ascore(sample)

    print(f"Query: {query}")

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['metadata']['title']}"
        )

    print(f"\nContext Precision: {precision:.3f}")
    print(f"Context Recall: {recall:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
