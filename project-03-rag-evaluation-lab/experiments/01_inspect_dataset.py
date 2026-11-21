from project_03_rag_evaluation_lab.dataset import load_qasper
from project_03_rag_evaluation_lab.evaluation_dataset import prepare_evaluation_question

dataset = load_qasper()

paper = dataset[0]
qas = paper["qas"]

print("Paper ID:", paper["id"])
print("Title:", paper["title"])
print("Number of questions:", len(qas["question"]))

for index, question in enumerate(qas["question"]):
    print(f"\nQuestion {index}: {question}")
    print("Question ID:", qas["question_id"][index])
    print("Annotations:", len(qas["answers"][index]["answer"]))
    
    
question_index = 0
question = qas["question"][question_index]
annotations = qas["answers"][question_index]["answer"]
print("\nQuestion:", question)

for index, annotation in enumerate(annotations, start=1):
    print(f"\nAnnotation {index}")
    print("Free-form answer:", annotation["free_form_answer"])
    print("Extractive spans:", annotation["extractive_spans"])
    print("Yes/No:", annotation["yes_no"])
    print("Unanswerable:", annotation["unanswerable"])
    print("Evidence:", annotation["evidence"])
    
    
result = prepare_evaluation_question(paper=paper, question_index=question_index)
print("Reference answers:", len(result.reference_answers))
print("Gold evidence:", len(result.gold_evidence))
print("Unanswerable:", result.is_unanswerable)