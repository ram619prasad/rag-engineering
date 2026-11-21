from pydantic import BaseModel

class EvaluationQuestion(BaseModel):
    question: str
    question_id: str
    paper_id: str
    is_unanswerable: bool
    reference_answers: list[str]
    gold_evidence: list[str]


def prepare_evaluation_question(
    paper: dict,
    question_index: int,
) -> EvaluationQuestion:
    qas = paper["qas"]

    question = qas["question"][question_index]
    question_id = qas["question_id"][question_index]
    annotations = qas["answers"][question_index]["answer"]

    reference_answers: list[str] = []
    gold_evidence: list[str] = []
    answerability_flags: list[bool] = []

    for annotation in annotations:
        free_form = annotation["free_form_answer"]

        if free_form: # Human written answers
            reference_answers.append(free_form)
        else:
            reference_answers.extend(annotation["extractive_spans"])

        gold_evidence.extend(annotation["evidence"])
        gold_evidence = list(dict.fromkeys(gold_evidence)) # Remove dupes

        answerability_flags.append(annotation["unanswerable"])

        # Don't silently combine disagreeing annotations
        if len(set(answerability_flags)) > 1:
            raise ValueError(
                f"Annotators disagree on answerability for {question_id}"
            )

    return EvaluationQuestion(
        question=question,
        question_id=question_id,
        paper_id=paper["id"],
        reference_answers=reference_answers,
        gold_evidence=gold_evidence,
        is_unanswerable=answerability_flags[0],
    )
    