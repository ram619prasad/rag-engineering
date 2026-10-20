
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class EvaluationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    relevant_chunk_ids: list[str]


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[EvaluationRecord]


def load_dataset() -> list[EvaluationRecord]:
    project_dir = Path(__file__).resolve().parents[2]
    dataset_path = project_dir / "datasets" / "evaluation_dataset.json"

    with dataset_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    # The JSON file is a list, so validate each record as a model.
    records = [
        EvaluationRecord.model_validate(record)
        for record in raw_data
    ]

    # Enforce uniqueness across the dataset.
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation record IDs must be unique.")

    return records


def main() -> None:
    records = load_dataset()

    print(f"Loaded {len(records)} evaluation records")
    print(f"First question: {records[0].question}")
    print(f"First relevant chunks: {records[0].relevant_chunk_ids}")


if __name__ == "__main__":
    main()
