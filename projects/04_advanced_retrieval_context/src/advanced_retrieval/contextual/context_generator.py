
import json

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from advanced_retrieval.retrievers.bm25_retriever import CATALOG_PATH

load_dotenv()

ROOT = CATALOG_PATH.parents[2]
DOCUMENTS_DIR = CATALOG_PATH.parent / "documents"
OUTPUT_PATH = CATALOG_PATH.parent / "contextual_sections.json"

# Start small. Change to None after inspecting the first 5.
MAX_NEW_SECTIONS = 434


def save_results(results: dict):
    OUTPUT_PATH.write_text(
        json.dumps(
            {"sections": list(results.values())},
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )


def generate_context(
    llm: ChatOpenAI,
    document_text: str,
    section_text: str,
) -> str:
    response = llm.invoke([
        SystemMessage(
            content=(
                "You generate concise context for engineering-document retrieval. "
                "Explain how the target section fits into its source document. "
                "Add only information that helps someone find or understand "
                "the section when it is read independently. "
                "Prioritize missing technical names, relationships, and scope. "
                "Avoid repeating details already explicit in the section. "
                "Use only facts supported by the document. "
                "Write at most 40 words. Return only the context."
            )
        ),
        HumanMessage(
            content=(
                f"<document>\n{document_text}\n</document>\n\n"
                f"<section>\n{section_text}\n</section>"
            )
        ),
    ])

    return str(response.content).strip()


def main():
    sections = json.loads(
        CATALOG_PATH.read_text(encoding="utf-8")
    )

    # Resume previously generated sections.
    if OUTPUT_PATH.exists():
        saved = json.loads(
            OUTPUT_PATH.read_text(encoding="utf-8")
        )

        results = {
            section["section_id"]: section
            for section in saved["sections"]
        }
    else:
        results = {}

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        timeout=60,
        max_retries=2,
    )

    document_cache = {}
    generated = 0

    for section in sections:
        section_id = section["section_id"]
        document_id = section["document_id"]

        if section_id in results:
            continue

        if (
            MAX_NEW_SECTIONS is not None
            and generated >= MAX_NEW_SECTIONS
        ):
            break

        document_path = (
            DOCUMENTS_DIR / f"{document_id}.md"
        )

        if document_id not in document_cache:
            document_cache[document_id] = (
                document_path.read_text(encoding="utf-8")
            )

        try:
            context = generate_context(
                llm=llm,
                document_text=document_cache[document_id],
                section_text=section["text"],
            )

            if not context:
                raise ValueError("Empty generated context")

            results[section_id] = {
                **section,
                "context": context,
                "contextual_text": (
                    f"{context}\n\n{section['text']}"
                ),
            }

            save_results(results)
            generated += 1

            print(
                f"[{len(results)}/{len(sections)}] "
                f"{section_id}"
            )

        except Exception as exc:
            print(f"Failed {section_id}: {exc}")

    print(
        f"\nSaved {len(results)}/{len(sections)} "
        f"contextualized sections to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
