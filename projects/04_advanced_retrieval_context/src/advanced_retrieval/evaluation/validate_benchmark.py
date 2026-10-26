
import json
import random
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "datasets" / "engineering_kb"

CATALOG_PATH = DATA_DIR / "section_catalog.json"
CANDIDATES_PATH = DATA_DIR / "benchmark_candidates.json"

DEV_PATH = DATA_DIR / "benchmark_dev_generated.json"
TEST_PATH = DATA_DIR / "benchmark_test_generated.json"

TARGETS = {
    "factual": 50,
    "comparison": 30,
    "multi_section": 25,
    "cross_document": 15,
}

DEV_TARGETS = {
    "factual": 12,
    "comparison": 8,
    "multi_section": 6,
    "cross_document": 4,
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.lower()).strip()


def validate(questions: list[dict], catalog: list[dict]):
    sections = {
        section["section_id"]: section
        for section in catalog
    }

    errors = []
    seen_questions = set()
    seen_ids = set()
    counts = Counter()

    for index, question in enumerate(questions, start=1):
        question_id = question.get(
            "question_id", f"row_{index}"
        )

        question_type = question.get("question_type")
        text = question.get("question", "")
        relevant_ids = question.get(
            "relevant_section_ids", []
        )

        counts[question_type] += 1

        if question_id in seen_ids:
            errors.append(f"{question_id}: duplicate ID")

        seen_ids.add(question_id)

        if question_type not in TARGETS:
            errors.append(
                f"{question_id}: invalid question type"
            )

        normalized = normalize_question(text)

        if not normalized:
            errors.append(
                f"{question_id}: empty question"
            )

        if normalized in seen_questions:
            errors.append(
                f"{question_id}: duplicate question"
            )

        seen_questions.add(normalized)

        if not relevant_ids:
            errors.append(
                f"{question_id}: no relevant sections"
            )

        if len(relevant_ids) != len(set(relevant_ids)):
            errors.append(
                f"{question_id}: duplicate section IDs"
            )

        for section_id in relevant_ids:
            if section_id not in sections:
                errors.append(
                    f"{question_id}: unknown section {section_id}"
                )

        if question_type in (
            "comparison",
            "multi_section",
            "cross_document",
        ) and len(set(relevant_ids)) < 2:
            errors.append(
                f"{question_id}: expected at least two sections"
            )

        if question_type == "cross_document":
            documents = {
                sections[sid]["document_id"]
                for sid in relevant_ids
                if sid in sections
            }

            if len(documents) < 2:
                errors.append(
                    f"{question_id}: requires multiple documents"
                )

    for question_type, expected in TARGETS.items():
        if counts[question_type] != expected:
            errors.append(
                f"{question_type}: expected {expected}, "
                f"found {counts[question_type]}"
            )

    return errors, counts


def split_benchmark(questions: list[dict]):
    rng = random.Random(42)

    grouped = {
        question_type: []
        for question_type in TARGETS
    }

    for question in questions:
        grouped[question["question_type"]].append(
            question
        )

    dev = []
    test = []

    for question_type, group in grouped.items():
        rng.shuffle(group)

        dev_count = DEV_TARGETS[question_type]

        dev.extend(group[:dev_count])
        test.extend(group[dev_count:])

    return dev, test


def save_json(path: Path, questions: list[dict]):
    path.write_text(
        json.dumps(
            {"questions": questions},
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )


def main():
    catalog = load_json(CATALOG_PATH)
    candidates = load_json(CANDIDATES_PATH)["questions"]

    errors, counts = validate(candidates, catalog)

    print("\n=== BENCHMARK VALIDATION ===")

    print(f"Documents: {len(set(s['document_id'] for s in catalog))}")
    print(f"Sections: {len(catalog)}")
    print(f"Questions: {len(candidates)}")

    for question_type in TARGETS:
        print(f"{question_type}: {counts[question_type]}")

    if errors:
        print("\nValidation FAILED:")
        for error in errors:
            print(f" - {error}")

        raise SystemExit(1)

    print("\nStructural validation PASSED")

    dev, test = split_benchmark(candidates)

    save_json(DEV_PATH, dev)
    save_json(TEST_PATH, test)

    print("\n=== DATASET SPLIT ===")
    print(f"Development questions: {len(dev)}")
    print(f"Test questions: {len(test)}")

    print("\nDevelopment distribution:")
    for question_type, count in Counter(
        q["question_type"] for q in dev
    ).items():
        print(f"  {question_type}: {count}")

    print("\nFiles:")
    print(DEV_PATH)
    print(TEST_PATH)

    print(
        "\nNOTE: Relevance labels are provisional "
        "until manually reviewed."
    )


if __name__ == "__main__":
    main()
