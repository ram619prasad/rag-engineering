from pydantic import BaseModel, ConfigDict, Field

class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    category: str = Field(min_length=1)
    source: str = Field(min_length=1)
