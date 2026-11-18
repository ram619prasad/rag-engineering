
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from project_02_citation_aware_rag.models import PipelineResult

console = Console()


def _table(title: str, *columns: str) -> Table:
    table = Table(title=title, show_lines=True, expand=True)
    for column in columns:
        table.add_column(column, overflow="fold")
    return table


def _verdict(verdict: str) -> Text:
    colors = {
        "supported": "green",
        "contradicted": "red",
        "insufficient_evidence": "yellow",
    }
    return Text(verdict.replace("_", " ").title(),
                style=colors.get(verdict, "white"))


def display_header() -> None:
    console.print(
        Panel.fit(
            "Project 02 — Citation-Aware RAG\n"
            "End-to-End Execution Report",
            title="RAG Architect Mastery",
            border_style="cyan",
        )
    )


def display_summary(result: PipelineResult) -> None:
    table = _table("1. Pipeline Summary", "Component", "Result")

    rows = [
        ("Paper", result.paper_title),
        ("Documents in collection", str(result.documents_indexed)),
        ("Newly indexed", str(result.newly_indexed)),
        ("Retrieved paragraphs", str(len(result.retrieved_documents))),
        ("Extracted claims", str(len(result.claims))),
    ]

    for name, value in rows:
        table.add_row(name, value)

    console.print(table)


def display_grounded_answer(result: PipelineResult) -> None:
    console.print(
        Panel(
            Text(
                f"Question: {result.question}\n\n"
                f"Answer: {result.grounded_answer.answer}\n\n"
                f"Answerable: {result.grounded_answer.is_answerable}"
            ),
            title="2. Grounded Answer",
            border_style="blue",
        )
    )


def display_retrieved_evidence(result: PipelineResult) -> None:
    table = _table(
        "3. Retrieved Evidence",
        "Rank", "Document ID", "Evidence Preview",
    )

    for rank, doc in enumerate(result.retrieved_documents, 1):
        preview = doc.page_content[:180]
        if len(doc.page_content) > 180:
            preview += "..."

        table.add_row(str(rank), str(doc.id), preview)

    console.print(table)


def display_claims(result: PipelineResult) -> None:
    table = _table(
        "4. Extracted Claims",
        "#", "Claim", "Candidate Citations",
    )

    for index, claim in enumerate(result.claims, 1):
        table.add_row(
            str(index),
            claim.text,
            ", ".join(claim.citation_ids) or "None",
        )

    console.print(table)


def display_original_verifications(result: PipelineResult) -> None:
    table = _table(
        "5. Original Citation Verification",
        "Claim", "Citation", "Verdict", "Reason",
    )

    for item in result.original_verifications:
        table.add_row(
            item.claim,
            ", ".join(item.citation_ids) or "None",
            _verdict(item.verdict),
            item.reason,
        )

    console.print(table)


def display_discovered_citations(result: PipelineResult) -> None:
    table = _table(
        "6. Supporting Evidence Discovery",
        "Claim", "Supporting Citation",
    )

    for item in result.discovered_verifications:
        table.add_row(
            item.claim,
            ", ".join(item.citation_ids),
        )

    console.print(table)


def display_evaluation(result: PipelineResult) -> None:
    table = _table("7. Citation Quality Evaluation", "Metric", "Score")

    metrics = [
        ("Original Citation Precision", result.original_citation_precision),
        ("Original Claim Coverage", result.original_claim_coverage),
        ("Discovered Claim Coverage", result.discovered_claim_coverage),
    ]

    for name, score in metrics:
        table.add_row(
            name,
            f"{score:.1%}" if result.claims else "N/A",
        )

    console.print(table)


def display_pipeline_report(result: PipelineResult) -> None:
    display_header()
    display_summary(result)
    display_grounded_answer(result)
    display_retrieved_evidence(result)
    display_claims(result)
    display_original_verifications(result)
    display_discovered_citations(result)
    display_evaluation(result)
    