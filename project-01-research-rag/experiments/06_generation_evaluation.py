from deepeval.test_case.llm_test_case import LLMTestCase
from deepeval.metrics.faithfulness.faithfulness import FaithfulnessMetric
from deepeval.metrics.answer_relevancy.answer_relevancy import AnswerRelevancyMetric
from project_01_research_rag.vector_store import create_vector_store
from project_01_research_rag.retrieval import retrieve_evidence
from datasets import load_dataset
from project_01_research_rag.generation import generate_answer

faithfulness = FaithfulnessMetric(threshold=0.7)
relevancy = AnswerRelevancyMetric(threshold=0.7)

vector_store = create_vector_store()

dataset = load_dataset("allenai/qasper", split="train")
paper = dataset[0]
qas = paper["qas"]
paper_id = paper["id"]


question_indices = [0, 2, 4, 6, 7]
for index in question_indices:
    question = qas["question"][index]

    documents = retrieve_evidence(
        vector_store=vector_store,
        question=question,
        paper_id=paper_id,
        k=5,
    )

    generated_answer = generate_answer(
        question=question,
        documents=documents,
    )
    
    print("QUESTION:", question)

    print("\nGENERATED ANSWER:")
    print(generated_answer.model_dump_json(indent=2))
    
    print("\nRETRIEVED EVIDENCE:")
    for rank, doc in enumerate(documents, start=1):
        print(f"\nRank {rank}: {doc.id}")
        print(doc.page_content[:500])

    test_case = LLMTestCase(
        input=question,
        actual_output=generated_answer.answer,
        retrieval_context=[
            doc.page_content for doc in documents
        ],
    )

    faithfulness.measure(test_case)
    relevancy.measure(test_case)

    print("Faithfulness:", faithfulness.score)
    print("Faithfulness reason:", faithfulness.reason)

    print("Answer relevancy:", relevancy.score)
    print("Answer relevancy reason:", relevancy.reason)