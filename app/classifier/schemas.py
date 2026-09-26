from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    transcript: str = Field(min_length=1)

