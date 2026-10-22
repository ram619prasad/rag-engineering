from pydantic import BaseModel, ConfigDict, Field


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