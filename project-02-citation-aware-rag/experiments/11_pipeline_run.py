from project_02_citation_aware_rag.pipeline import run_pipeline
from project_02_citation_aware_rag.reporting import display_pipeline_report

result = run_pipeline(
    question="What is the seed lexicon?",
    paper_index=0,
    k=5,
)


display_pipeline_report(result)