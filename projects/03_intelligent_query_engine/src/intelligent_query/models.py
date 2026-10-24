from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    title: str
    text: str
    score: float | None = None
    retrieval_method: str


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    title: str
    text: str
    chunk_index: int = Field(ge=0)

    source: str = "scifact" # metadata
    content_hash: str # metadata to avoid reembedding
    
class QueryClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_type: Literal[
        "exact_match",
        "factual",
        "explanatory",
        "comparison",
        "troubleshooting",
        "multi_hop",
    ]

    retrieval_strategy: Literal[
        "bm25",
        "semantic",
        "hybrid",
    ]

    reason: str = Field(
        description="Brief explanation for the classification."
    )
    
    
class DecomposedQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subquestions: list[str] = Field(
        description=(
            "Independent, focused questions that collectively "
            "address the original complex question."
        )
    )
