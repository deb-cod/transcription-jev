from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    transcript: str = Field(min_length=1)
    backend: str | None = None
    model: str | None = None
