
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = ROOT / "datasets" / "engineering_kb"
CATALOG_PATH = DATASET_DIR / "section_catalog.json"
OUTPUT_PATH = DATASET_DIR / "benchmark_candidates.json"

TARGETS = {
    "factual": 50,
    "comparison": 30,
    "multi_section": 25,
    "cross_document": 15,
}

SEED = 42


class GeneratedQuestion(BaseModel):
    question: str | None = Field(
        description=(
            "One concise engineering question, or null if "
            "the supplied sections cannot support a meaningful question"
        )
    )


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(questions: list[dict]):
    OUTPUT_PATH.write_text(
        json.dumps(
            {"questions": questions},
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )


def domain(section: dict) -> str:
    return section["document_id"].split("_", 1)[0]


def keywords(text: str) -> set[str]:
    stopwords = {
        "the", "and", "for", "with", "from", "into",
        "that", "this", "when", "using", "used",
        "how", "what", "are", "can", "should",
        "overview", "example", "examples",
    }

    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) >= 4 and word not in stopwords
    }


def similarity(a: dict, b: dict) -> int:
    # Used only for selecting candidate evidence pairs.
    # It does NOT prove the sections support a question.
    a_text = " ".join(a["heading_path"]) + " " + a["text"][:800]
    b_text = " ".join(b["heading_path"]) + " " + b["text"][:800]

    return len(keywords(a_text) & keywords(b_text))


def prepare_groups(catalog: list[dict]) -> dict:
    rng = random.Random(SEED)

    sections = [
        section for section in catalog
        if section["heading_path"]
        and 300 <= len(section["text"]) <= 6000
    ]

    # Balance factual questions across domains.
    by_domain = defaultdict(list)

    for section in sections:
        by_domain[domain(section)].append(section)

    for group in by_domain.values():
        rng.shuffle(group)

    factual = []
    names = sorted(by_domain)

    while any(by_domain.values()):
        for name in names:
            if by_domain[name]:
                factual.append([by_domain[name].pop()])

    same_document = []
    cross_document = []

    for i, first in enumerate(sections):
        for second in sections[i + 1:]:
            score = similarity(first, second)

            if first["document_id"] == second["document_id"]:
                if score >= 2:
                    same_document.append((score, first, second))

            elif domain(first) == domain(second):
                if score >= 3:
                    cross_document.append((score, first, second))

    rng.shuffle(same_document)
    rng.shuffle(cross_document)

    same_document.sort(key=lambda x: x[0], reverse=True)
    cross_document.sort(key=lambda x: x[0], reverse=True)

    def pairs(items):
        return [
            [first, second]
            for _, first, second in items
        ]

    return {
        "factual": factual,
        "comparison": pairs(same_document),
        "multi_section": pairs(same_document),
        "cross_document": pairs(cross_document),
    }


def generate_question(llm, sections: list[dict], question_type: str):
    context = "\n\n".join(
        f"SECTION ID: {s['section_id']}\n"
        f"DOCUMENT: {s['document_id']}\n"
        f"HEADING: {' > '.join(s['heading_path'])}\n"
        f"CONTENT:\n{s['text']}"
        for s in sections
    )

    rules = {
        "factual": (
            "Ask one specific factual question from the source."
        ),
        "comparison": (
            "Ask a meaningful question comparing concepts or "
            "approaches covered by BOTH supplied sections."
        ),
        "multi_section": (
            "Ask a question that genuinely requires evidence "
            "from BOTH supplied sections."
        ),
        "cross_document": (
            "Ask a natural question requiring evidence from "
            "BOTH documents."
        ),
    }

    response = llm.invoke([
        (
            "system",
            "Generate one realistic engineering retrieval question. "
            "Use only supplied evidence. "
            "Keep it concise and self-contained, ideally under 25 words. "
            "Do not invent facts or mention section IDs. "
            "Avoid unrelated compound questions. "
            "If the source material cannot support the requested "
            "question type, return null. "
            + rules[question_type]
        ),
        ("human", context),
    ])

    if not response.question:
        return None

    question = response.question.strip()

    if not question.endswith("?"):
        return None

    return {
        "question": question,
        "question_type": question_type,
        "relevant_section_ids": [
            s["section_id"] for s in sections
        ],
        "status": "needs_review",
    }


def main():
    catalog = load_json(CATALOG_PATH)
    groups = prepare_groups(catalog)

    if OUTPUT_PATH.exists():
        candidates = load_json(OUTPUT_PATH)["questions"]
    else:
        candidates = []

    # Existing candidates are retained.
    counts = Counter(
        q["question_type"] for q in candidates
    )

    existing_keys = {
        (
            q["question_type"],
            tuple(q["relevant_section_ids"]),
        )
        for q in candidates
    }

    existing_text = {
        re.sub(r"\s+", " ", q["question"].lower()).strip()
        for q in candidates
    }

    next_id = max(
        (
            int(match.group(1))
            for q in candidates
            if (match := re.fullmatch(
                r"candidate_(\d+)",
                q.get("question_id", ""),
            ))
        ),
        default=0,
    ) + 1

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        timeout=60,
        max_retries=2,
    ).with_structured_output(GeneratedQuestion)

    for question_type, target in TARGETS.items():
        print(f"\n=== GENERATING {question_type.upper()} ===")

        for sections in groups[question_type]:
            if counts[question_type] >= target:
                break

            section_ids = tuple(
                s["section_id"] for s in sections
            )
            key = (question_type, section_ids)

            if key in existing_keys:
                continue

            # Don't retry the same pair within this run.
            existing_keys.add(key)

            try:
                question = generate_question(
                    llm,
                    sections,
                    question_type,
                )

                if question is None:
                    continue

                normalized = re.sub(
                    r"\s+",
                    " ",
                    question["question"].lower(),
                ).strip()

                if normalized in existing_text:
                    continue

                question["question_id"] = (
                    f"candidate_{next_id:04d}"
                )
                next_id += 1

                candidates.append(question)
                existing_text.add(normalized)
                counts[question_type] += 1

                # Checkpoint after every successful question.
                save(candidates)

                print(
                    f"[{counts[question_type]}/{target}] "
                    f"{question['question']}"
                )

            except Exception as exc:
                print(f"Generation failed: {exc}")

        if counts[question_type] < target:
            print(
                f"Insufficient valid candidates for {question_type}: "
                f"{counts[question_type]}/{target}"
            )

    print("\n=== FINAL SUMMARY ===")

    for question_type, target in TARGETS.items():
        print(
            f"{question_type}: "
            f"{counts[question_type]}/{target}"
        )

    print(f"Total candidates: {len(candidates)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
